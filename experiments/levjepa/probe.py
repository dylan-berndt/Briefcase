"""
Frozen-feature probe for LeVJEPA snapshots and for the control (the go/no-go in CLAUDE.md).

Embeds the MyFonts fonts with a frozen encoder, trains the same small MLP probe as the earlier critical-review probe
(Dropout 0.2 - Linear(d, 1024) - ReLU - Dropout 0.3 - Linear(1024, tags), AdamW 1e-3 / wd 1e-2, 40 epochs, batch 256,
standardized features) on the official MyFonts train split, and reports on the test split: ICCV mAP + NDCG
(single-300 / single-full / multi), AMT accuracy, top-50 ROC/PR-AUC. Metric code is finetuneTags.py's.

  LeVJEPA snapshot : --ckpt checkpoints/pretrain/levjepa-d256-e200-snapshots/step5000
      --mode full    the whole font in one pass, every patch of every present glyph, no token dropping, CLS output
                     (the embedding the design calls for; 1,872 tokens for a 52-glyph font)
      --mode sparse  mean CLS over --sparseViews random ink-biased global views with the training keep ratio
                     (what the encoder saw while training; useful to tell "sparse->full shift" from "bad features")
  control          : --control checkpoints/pretrain/best [--glyphs lower|all]
                     mean of the L2-normalized per-glyph CLS over 26 lowercase or all 52 glyphs, same cache and probe
  random encoder   : --ckpt <dir of an untrained encoder> gives the floor.

Safe to run while training: the process caps its own CUDA allocator (--vramFraction), refuses to start if the card has
less than --minFreeGB free, uses small batches, and the probe head trains on the CPU. It will slow training while it embeds.
Fonts are the ones in dataset/levjepa with >= --minGlyphs glyphs, so the test set is a little smaller than the official
1,877 fonts.

    python -u experiments/levjepa/probe.py --ckpt <snapshot dir>
    python -u experiments/levjepa/probe.py --control checkpoints/pretrain/best --glyphs lower
    python -u experiments/levjepa/probe.py --follow checkpoints/pretrain/levjepa-d256-e200-snapshots   # probe each new snapshot
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import types
from collections import Counter

_scriptDir = os.path.dirname(os.path.abspath(__file__))
_repoRoot = os.path.dirname(os.path.dirname(_scriptDir))
os.chdir(_repoRoot)
for _p in (_scriptDir, _repoRoot):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train as levjepa          # Encoder, ViewSampler (experiments/levjepa/train.py)
import finetuneTags as ft        # readList, loadTags, top50Metrics, iccvMetrics, amtAccuracy (repo root)
from utils.vit import ViT

RESULTS = os.path.join("results", "levjepaProbe")


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="LeVJEPA encoder dir (checkpoint.pt + config.json)")
    p.add_argument("--control", help="old ViT checkpoint dir, e.g. checkpoints/pretrain/best")
    p.add_argument("--glyphs", choices=["lower", "all"], default="lower", help="control: which glyphs to pool")
    p.add_argument("--bitmap64", action="store_true", help="control: read 64px glyphs from dataset/smallimage_64")
    p.add_argument("--mode", choices=["full", "sparse"], default="full")
    p.add_argument("--sparseViews", type=int, default=8)
    p.add_argument("--keep", type=float, default=0.10)
    p.add_argument("--inkFrac", type=float, default=0.75)
    p.add_argument("--inkThreshold", type=float, default=0.1)
    p.add_argument("--data", default=os.path.join("dataset", "levjepa"))
    p.add_argument("--datasetDir", default="dataset")
    p.add_argument("--minGlyphs", type=int, default=26)
    p.add_argument("--maxFonts", type=int, default=None, help="subsample fonts per split (cheap probe)")
    p.add_argument("--batch", type=int, default=8, help="fonts per forward in full mode (64 in sparse, 16 for control)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vramFraction", type=float, default=0.30)
    p.add_argument("--minFreeGB", type=float, default=2.5)
    p.add_argument("--tag", default=None, help="output name (default derived from the checkpoint dir)")
    p.add_argument("--wandb", action="store_true", help="log to a separate wandb run under LeVJEPA/probe/")
    p.add_argument("--follow", help="snapshots dir: probe every new stepN folder as it appears")
    p.add_argument("--total", type=int, default=59000, help="--follow exits after this step")
    p.add_argument("--followPoll", type=float, default=300.0)
    return p.parse_args()


# ------------------------------------------------------------------ data

class Cache:
    def __init__(self, dataDir):
        self.glyphs = np.load(os.path.join(dataDir, "glyphs.npy"), mmap_mode="r")
        self.present = np.load(os.path.join(dataDir, "present.npy"))
        with open(os.path.join(dataDir, "meta.json"), encoding="utf8") as f:
            meta = json.load(f)
        self.rowOf = {n: i for i, (n, s) in enumerate(zip(meta["names"], meta["sources"])) if s == "myfonts"}
        self.names = meta["names"]


class Bitmap64:
    """MyFonts glyphs from dataset/smallimage_64 (loadRochesterImage at fontSize 43 -> 64px, the same MyFonts
    pipeline as the 48px cache), indexed like Cache: glyphs[rows] -> uint8 [B, 52, 64, 64], present [rows] -> bool."""

    LETTERS = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, cache, folder=os.path.join("dataset", "smallimage_64")):
        self.cache, self.folder = cache, folder
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(8)

    def _font(self, row):
        name = self.cache.names[row]
        g = np.zeros((52, 64, 64), np.uint8)
        pr = np.zeros(52, bool)
        for i, ch in enumerate(self.LETTERS):
            path = os.path.join(self.folder, f"{name} {ch.lower()}{'l' if ch.islower() else 'u'}.bmp")
            if os.path.exists(path):
                from PIL import Image
                a = np.asarray(Image.open(path).convert("L"))
                if a.shape == (64, 64):
                    g[i], pr[i] = a, True
        return g, pr

    def fetch(self, rows):
        out = list(self.pool.map(self._font, list(rows)))
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def splitRows(cache, datasetDir, minGlyphs, maxFonts, seed):
    """Official MyFonts train/val/test lists -> {split: (names, rows)} for fonts present in the cache."""
    rng = np.random.RandomState(seed)
    out = {}
    for split in ("train", "val", "test"):
        names = [n for n in ft.readList(os.path.join(datasetDir, "fontset", f"{split}set"))
                 if n in cache.rowOf and cache.present[cache.rowOf[n]].sum() >= minGlyphs]
        if maxFonts and len(names) > maxFonts:
            names = sorted(rng.choice(names, maxFonts, replace=False).tolist())
        out[split] = (names, np.array([cache.rowOf[n] for n in names]))
    return out


# ------------------------------------------------------------------ embedding

def buildTokens(x, letters):
    """x float [B, G, 48, 48] (glyph = 1), letters long [G] -> tokens [B, G*36, 64], coords [B, G*36, 3]."""
    B, G = x.shape[:2]
    p = x.view(B, G, 6, 8, 6, 8).permute(0, 1, 2, 4, 3, 5).reshape(B, G * 36, 64)
    pos = torch.arange(36, device=x.device).repeat(G)
    coords = torch.stack([letters.repeat_interleave(36).float(), (pos // 6).float(), (pos % 6).float()], dim=-1)
    return p, coords.unsqueeze(0).expand(B, -1, -1)


def loadEncoder(ckptDir, device):
    with open(os.path.join(ckptDir, "config.json")) as f:
        m = json.load(f)["model"]
    enc = levjepa.Encoder(m["dim"], m["depth"], m["heads"], m["mlpRatio"])
    enc.load_state_dict(torch.load(os.path.join(ckptDir, "checkpoint.pt"), map_location="cpu", weights_only=True))
    return enc.to(device).eval(), m["dim"]


@torch.no_grad()
def embedFull(enc, dim, cache, rows, device, batch):
    feats = np.zeros((len(rows), dim), np.float32)
    isFull = cache.present[rows].all(1)
    full, partial = np.where(isFull)[0], np.where(~isFull)[0]
    letters = torch.arange(levjepa.NUM_GLYPHS, device=device)
    for s in range(0, len(full), batch):
        ids = full[s:s + batch]
        x = torch.from_numpy(cache.glyphs[rows[ids]]).to(device).float() / 255.0
        tokens, coords = buildTokens(x, letters)
        feats[ids] = enc(tokens, coords).cpu().numpy()
    for i in partial:
        keep = np.where(cache.present[rows[i]])[0]
        x = torch.from_numpy(cache.glyphs[rows[i]][keep])[None].to(device).float() / 255.0
        tokens, coords = buildTokens(x, torch.tensor(keep, device=device))
        feats[i] = enc(tokens, coords).cpu().numpy()[0]
    return feats


@torch.no_grad()
def embedSparse(enc, dim, cache, rows, device, batch, args):
    cfg = types.SimpleNamespace(keep=args.keep, inkFrac=args.inkFrac, inkThreshold=args.inkThreshold,
                                cropPatches=3, views=0)
    sampler = levjepa.ViewSampler(cfg, device)
    torch.manual_seed(args.seed)
    feats = np.zeros((len(rows), dim), np.float32)
    for s in range(0, len(rows), batch):
        ids = np.arange(s, min(s + batch, len(rows)))
        g = torch.from_numpy(cache.glyphs[rows[ids]]).to(device)
        present = torch.from_numpy(cache.present[rows[ids]]).to(device)
        patches, ink = sampler.patchify(g)
        acc = 0
        for _ in range(args.sparseViews):
            tokens, coords = sampler.globalView(patches, ink, present)
            acc = acc + enc(tokens, coords)
        feats[ids] = (acc / args.sparseViews).cpu().numpy()
    return feats


@torch.no_grad()
def embedControl(ckptDir, cache, rows, sel, device, batch, bitmap64=None):
    model, _ = ViT.load(ckptDir)
    model = model.to(device).eval()
    dim = model.config.embedDim
    feats = np.zeros((len(rows), dim), np.float32)
    for s in range(0, len(rows), batch):
        ids = np.arange(s, min(s + batch, len(rows)))
        if bitmap64 is None:
            g, pr = cache.glyphs[rows[ids]], cache.present[rows[ids]]
        else:
            g, pr = bitmap64.fetch(rows[ids])
        x = torch.from_numpy(np.ascontiguousarray(g[:, sel])).to(device).float() / 255.0     # [B, G, S, S]
        present = torch.from_numpy(pr[:, sel]).to(device).float()
        B, G, S = x.shape[:3]
        assert S == model.config.imageSize, f"{S}px glyphs for a {model.config.imageSize}px model (use --bitmap64?)"
        _, c = model(x.reshape(B * G, S, S).unsqueeze(-1))
        c = F.normalize(c, dim=-1).view(B, G, -1)
        feats[ids] = ((c * present.unsqueeze(-1)).sum(1) / present.sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
    return feats


# ------------------------------------------------------------------ probe

def trainProbe(Xtr, Ytr, epochs, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Dropout(0.2), nn.Linear(Xtr.shape[1], 1024), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(1024, Ytr.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 256):
            b = perm[s:s + 256]
            loss = F.binary_cross_entropy_with_logits(net(Xtr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def evaluateProbe(feats, splits, datasetDir, epochs, seed):
    torch.set_num_threads(4)
    tags = {n: set(open(os.path.join(datasetDir, "taglabel", n)).read().split())
            for split in splits.values() for n in split[0]}
    counts = Counter(t for n in splits["train"][0] for t in tags[n])
    vocab = [t for t, _ in counts.most_common()]
    vocabIndex = {t: i for i, t in enumerate(vocab)}
    top50Idx = list(range(min(50, len(vocab))))

    def targets(names):
        y = np.zeros((len(names), len(vocab)), np.float32)
        for i, n in enumerate(names):
            for t in tags[n]:
                if t in vocabIndex:
                    y[i, vocabIndex[t]] = 1
        return y

    Xtr = feats["train"]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    std = lambda a: torch.tensor(((a - mu) / sd).astype(np.float32))
    net = trainProbe(std(Xtr), torch.tensor(targets(splits["train"][0])), epochs, seed)
    with torch.no_grad():
        logits = {k: net(std(feats[k])).numpy() for k in ("val", "test")}

    testNames = splits["test"][0]
    testTags = [tags[n] for n in testNames]
    single = [[t] for t in ft.readList(os.path.join(datasetDir, "myfonts-testset", "singletag-test"))]
    multi = [q.split("&&&") for q in ft.readList(os.path.join(datasetDir, "myfonts-testset", "multitag-test"))]
    top300 = set(vocab[:300])
    valRoc, valPr = ft.top50Metrics(logits["val"], targets(splits["val"][0]), top50Idx)
    roc, pr = ft.top50Metrics(logits["test"], targets(testNames), top50Idx)
    return {
        "fonts": {k: len(v[0]) for k, v in splits.items()}, "tags": len(vocab),
        "val": {"top50RocAuc": valRoc, "top50PrAuc": valPr},
        "top50RocAuc": roc, "top50PrAuc": pr,
        "iccvSingle300": ft.iccvMetrics(logits["test"], testTags, [q for q in single if q[0] in top300], vocabIndex),
        "iccvSingleFull": ft.iccvMetrics(logits["test"], testTags, single, vocabIndex),
        "iccvMulti": ft.iccvMetrics(logits["test"], testTags, multi, vocabIndex),
        "amt": ft.amtAccuracy(logits["test"], testNames, vocabIndex, datasetDir),
    }


def summary(tag, r):
    return (f"[{tag}] fonts {r['fonts']} | top-50 ROC-AUC {r['top50RocAuc']:.4f} PR-AUC {r['top50PrAuc']:.4f} | "
            f"ICCV mAP single-300 {r['iccvSingle300']['mAP']:.2f} full {r['iccvSingleFull']['mAP']:.2f} "
            f"multi {r['iccvMulti']['mAP']:.2f} | NDCG {r['iccvSingle300']['NDCG']:.1f}/{r['iccvSingleFull']['NDCG']:.1f}/"
            f"{r['iccvMulti']['NDCG']:.1f} | AMT {r['amt']['accuracy']:.3f} (n={r['amt']['groups']})")


# ------------------------------------------------------------------ driver

def guardGpu(args):
    if args.device != "cuda":
        return
    free, total = torch.cuda.mem_get_info()
    if free < args.minFreeGB * 2 ** 30:
        raise SystemExit(f"only {free / 2 ** 30:.2f} GiB free on the GPU (< {args.minFreeGB}), not starting")
    torch.cuda.set_per_process_memory_fraction(args.vramFraction)


def runProbe(args, cache):
    tag = args.tag
    if args.control:
        tag = tag or f"control_{os.path.basename(os.path.normpath(args.control))}_{args.glyphs}"
    else:
        base = os.path.basename(os.path.normpath(args.ckpt))
        parent = os.path.basename(os.path.dirname(os.path.normpath(args.ckpt)))
        tag = tag or f"{parent}_{base}_{args.mode}"
    os.makedirs(RESULTS, exist_ok=True)
    splits = splitRows(cache, args.datasetDir, args.minGlyphs, args.maxFonts, args.seed)
    started = time.time()
    feats = {}
    if args.control:
        sel = np.arange(26) if args.glyphs == "lower" else np.arange(52)
        for k, (_, rows) in splits.items():
            feats[k] = embedControl(args.control, cache, rows, sel, args.device, args.batch if args.batch > 8 else 16,
                                    Bitmap64(cache) if args.bitmap64 else None)
    else:
        enc, dim = loadEncoder(args.ckpt, args.device)
        for k, (_, rows) in splits.items():
            feats[k] = (embedFull(enc, dim, cache, rows, args.device, args.batch) if args.mode == "full"
                        else embedSparse(enc, dim, cache, rows, args.device, 64, args))
        del enc
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else 0.0
    if args.device == "cuda":
        torch.cuda.empty_cache()
    embedSeconds = time.time() - started
    result = evaluateProbe(feats, splits, args.datasetDir, args.epochs, args.seed)
    result.update({"tag": tag, "args": vars(args), "embedSeconds": embedSeconds, "peakGpuGiB": peak,
                   "totalSeconds": time.time() - started})
    with open(os.path.join(RESULTS, tag + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    print(summary(tag, result), flush=True)
    print(f"    embed {embedSeconds:.0f}s, total {time.time() - started:.0f}s, peak GPU {peak:.2f} GiB allocated", flush=True)
    return result


def logWandb(args, result, step):
    import wandb
    runName = (os.path.basename(os.path.normpath(args.follow)) if args.follow
               else os.path.basename(os.path.dirname(os.path.normpath(args.ckpt))))
    run = wandb.init(entity="dylanberndt123-missouri-state-university", project="Briefcase",
                     name="levjepa-probe", id="levjepa-probe-" + runName, resume="allow")
    run.log({"LeVJEPA/probe/top50 ROC-AUC": result["top50RocAuc"], "LeVJEPA/probe/top50 PR-AUC": result["top50PrAuc"],
             "LeVJEPA/probe/mAP single300": result["iccvSingle300"]["mAP"],
             "LeVJEPA/probe/mAP singleFull": result["iccvSingleFull"]["mAP"],
             "LeVJEPA/probe/mAP multi": result["iccvMulti"]["mAP"], "LeVJEPA/probe/AMT": result["amt"]["accuracy"],
             "LeVJEPA/probe/step": step}, step=step)
    run.finish()


def main():
    args = parseArgs()
    assert args.follow or args.ckpt or args.control, "give --ckpt, --control or --follow"
    guardGpu(args)
    cache = Cache(args.data)
    if not args.follow:
        result = runProbe(args, cache)
        if args.wandb:
            m = re.search(r"step(\d+)", args.ckpt or "")
            logWandb(args, result, int(m.group(1)) if m else 0)
        return

    seen = set()
    while True:
        for d in sorted(glob.glob(os.path.join(args.follow, "step*")), key=lambda s: int(re.search(r"step(\d+)", s).group(1))):
            if d.endswith(".tmp") or not os.path.exists(os.path.join(d, "config.json")) or d in seen:
                continue
            step = int(re.search(r"step(\d+)", d).group(1))
            args.ckpt, args.tag = d, f"{os.path.basename(args.follow).replace('-snapshots', '')}_step{step}_{args.mode}"
            if not os.path.exists(os.path.join(RESULTS, args.tag + ".json")):
                try:
                    result = runProbe(args, cache)
                    if args.wandb:
                        logWandb(args, result, step)
                except Exception as error:
                    print(f"probe of {d} failed: {error!r}", flush=True)
                    continue
            seen.add(d)
            if step >= args.total:
                return
        time.sleep(args.followPoll)


if __name__ == "__main__":
    main()
