# End-to-end tag finetuning of the pretrained ViT on the MyFonts (ICCV 2019) tags.
#
# retrieval.py trains a head on FROZEN embeddings. This script unfreezes the backbone too:
# a few head-only epochs first, then the whole ViT at a much lower learning rate than the head.
#
# Data: MyFonts only, the dataset's own official splits (fontset/trainset|valset|testset),
# lowercase a-z glyphs rendered exactly like utils/loaders/myfonts.loadRochesterImage (per-glyph
# scale-to-fit, then the uint8 bmp round trip the normal pipeline goes through). Rendered once and
# cached to <datasetDir>/finetuneTagsCache_*.npy.
#
# Backbone: `backbone` in the config is either a utils.vit.ViT checkpoint (e.g. checkpoints/pretrain/best) or a
# LeVJEPA run/snapshot (experiments/levjepa; detected from its config.json, model.type == "levjepa").
# Font representation: ViT -- mean of the L2-normalized per-glyph CLS vectors, the same pooling
# utils/embeddings.generateEmbeddings uses for embeddings/all.json; LeVJEPA -- the encoder's CLS over all of the font's
# glyphs at once (the embedding experiments/levjepa/probe.py scores). LeVJEPA needs fontSize 32 (48px). Training samples
# glyphsPerFont random letters per font per step; evaluation uses all 26.
#
# No rotation/aspect augmentation: tags describe slant and width, so those augmentations would
# randomize the very attributes being predicted. shiftAug adds a small random translation only.
#
# Metrics (all on the official splits):
#   - top-50 macro ROC-AUC / PR-AUC, the music auto-tagging protocol (Won et al. 2020). Val
#     top-50 ROC-AUC is the early-stopping criterion.
#   - ICCV MyFonts-test protocol: single-tag (top-300 / full) and multi-tag mAP + NDCG.
#   - AMT-test accuracy (human "which of 3 fonts best fits this tag").
# Test metrics are reported twice from the same code: at the end of the head-only phase
# ("frozenBaseline") and for the best finetuned epoch.
#
#   python finetuneTags.py                                  # full run (GPU), settings in configs/finetuneTags.json
#   python finetuneTags.py --maxFonts 300 --maxEpochs 2     # smoke test

import argparse
import csv
import json
import math
import os
import random
import zlib
from collections import Counter
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.config import loadRunConfig
from utils.vit import ViT
from utils.loaders.myfonts import loadRochesterImage

LETTERS = "abcdefghijklmnopqrstuvwxyz"
device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArgs():
    """Every setting that defines a run is in the config file (configs/finetuneTags.json: data, model, training,
    output; minTagCount 1 keeps every train tag so the ICCV query sets stay comparable, shiftAug is the max random
    translation in pixels, 0 = off; backbone is a ViT or LeVJEPA checkpoint dir; autocast = fp16 mixed precision on
    CUDA, which measured ~5x slower per step for LeVJEPA training on the GTX 1660 SUPER, so try false there).
    The flags only change how this process runs."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join("configs", "finetuneTags.json"))
    p.add_argument("--maxFonts", type=int, default=None, help="subsample fonts per split, for smoke tests")
    p.add_argument("--maxEpochs", type=int, default=None, help="stop after this many epochs, for smoke tests")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--evalOnly", default=None, metavar="RUN_DIR",
                   help="skip training: load checkpoint.pt + head.pt from an interrupted run's directory, then run the "
                        "final test report and write the run's metadata files (no frozen baseline)")
    args = p.parse_args()
    return loadRunConfig(args.config, args)


# ---------------------------------------------------------------- data

def readList(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def renderFont(args):
    fontDir, name, fontSize = args
    glyphs = []
    for letter in LETTERS:
        path = os.path.join(fontDir, f"{name}_{letter}.png")
        if not os.path.exists(path):
            return name, None
        _, canvas = loadRochesterImage((path, fontSize))
        if canvas is None:
            return name, None
        # same truncating uint8 conversion loadMyFontsImagePaths uses when it writes the .bmp cache
        glyphs.append((canvas * 255).astype(np.uint8))
    return name, np.stack(glyphs)


def loadGlyphCache(datasetDir, names, fontSize, workers):
    tag = f"{len(names)}_{zlib.crc32(chr(10).join(names).encode())}"
    cachePath = os.path.join(datasetDir, f"finetuneTagsCache_{fontSize}_{tag}.npz")
    if os.path.exists(cachePath):
        data = np.load(cachePath, allow_pickle=False)
        return list(data["names"]), data["glyphs"]

    fontDir = os.path.join(datasetDir, "fontimage")
    kept, glyphs = [], []
    tasks = [(fontDir, name, fontSize) for name in names]
    with Pool(workers) as pool:
        for i, (name, array) in enumerate(pool.imap(renderFont, tasks, chunksize=32)):
            if array is not None:
                kept.append(name)
                glyphs.append(array)
            if i % 500 == 0:
                print(f"\rRendering glyphs: {i + 1}/{len(names)}", end="")
    print()
    glyphs = np.stack(glyphs)
    np.savez(cachePath, names=np.array(kept), glyphs=glyphs)
    return kept, glyphs


def loadTags(datasetDir, names):
    return {n: set(open(os.path.join(datasetDir, "taglabel", n)).read().split()) for n in names}


# ---------------------------------------------------------------- model

def clsFeatures(vit, x):
    """ViT CLS token without the reconstruction head. x: [N, H, W]."""
    x = vit.patching(x.unsqueeze(1))
    x = torch.cat([vit.clsToken.expand(x.shape[0], -1, -1), x], dim=1)
    return vit.transformer(x)[:, 0]


_levjepa = None


def levjepaModule():
    """experiments/levjepa/train.py (Encoder, GRID, PATCH, PER_GLYPH), loaded once by path."""
    global _levjepa
    if _levjepa is None:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "levjepa", "train.py")
        spec = importlib.util.spec_from_file_location("levjepaTrain", path)
        _levjepa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_levjepa)
    return _levjepa


def loadBackbone(path):
    """A pretrained backbone directory (checkpoint.pt + config.json) -> (module, kind, saveConfig(outDir)).
    kind is "levjepa" when config.json says model.type == "levjepa" (experiments/levjepa/train.py runs and their
    snapshots), else "vit" (utils.vit.ViT, e.g. checkpoints/pretrain/best)."""
    with open(os.path.join(path, "config.json")) as f:
        saved = json.load(f)
    if saved.get("model", {}).get("type") == "levjepa":
        m = saved["model"]
        encoder = levjepaModule().Encoder(m["dim"], m["depth"], m["heads"], m["mlpRatio"])
        encoder.load_state_dict(torch.load(os.path.join(path, "checkpoint.pt"), map_location="cpu", weights_only=True))

        def saveConfig(outDir):
            with open(os.path.join(outDir, "config.json"), "w") as f:
                json.dump(saved, f, indent=1)   # unchanged, so experiments/levjepa/embed.py and probe.py load the run
        return encoder, "levjepa", saveConfig
    vit, vitConfig = ViT.load(path)
    return vit, "vit", lambda outDir: vitConfig.save(os.path.join(outDir, "config.json"))


class FontTagger(nn.Module):
    """Tag head on a pretrained backbone. kind "vit": per-glyph ViT CLS, L2-normalized and averaged (the all.json
    pooling). kind "levjepa": the LeVJEPA encoder reads all of a font's given glyphs as one token sequence, each patch
    carrying (letter, row, col) rotary coordinates, and its CLS output is the font feature (the embedding probe.py
    evaluates)."""

    def __init__(self, backbone, numTags, hidden, kind="vit"):
        super().__init__()
        self.backbone, self.kind = backbone, kind
        dim = backbone.norm.normalized_shape[0] if kind == "levjepa" else backbone.config.embedDim
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.2),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, numTags),
        )

    def fontFeatures(self, glyphs, letters=None):
        """glyphs: [B, G, H, W]; letters: long [B, G] index of each glyph in a-z (default 0..G-1) -> [B, D]."""
        B, G, H, W = glyphs.shape
        if self.kind == "vit":
            cls = clsFeatures(self.backbone, glyphs.reshape(B * G, H, W))
            return F.normalize(cls, dim=-1).reshape(B, G, -1).mean(dim=1)
        lj = levjepaModule()
        assert H == W == lj.GRID * lj.PATCH, f"LeVJEPA backbones take {lj.GRID * lj.PATCH}px glyphs, got {H}px"
        if letters is None:
            letters = torch.arange(G, device=glyphs.device).expand(B, -1)
        tokens = glyphs.reshape(B, G, lj.GRID, lj.PATCH, lj.GRID, lj.PATCH).permute(0, 1, 2, 4, 3, 5) \
            .reshape(B, G * lj.PER_GLYPH, lj.PATCH * lj.PATCH)
        pos = torch.arange(lj.PER_GLYPH, device=glyphs.device).repeat(G)
        coords = torch.stack([letters.repeat_interleave(lj.PER_GLYPH, dim=1).float(),
                              (pos // lj.GRID).float().expand(B, -1), (pos % lj.GRID).float().expand(B, -1)], -1)
        return self.backbone(tokens, coords)

    def forward(self, glyphs, letters=None):
        return self.head(self.fontFeatures(glyphs, letters))


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a trainable low-rank update: y = base(x) + (alpha / rank) * B(A(x)). B starts at zero, so
    the wrapped model begins identical to the pretrained one."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.loraA = nn.Parameter(torch.empty(rank, base.in_features))
        self.loraB = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.loraA, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.loraA), self.loraB) * self.scale

    def delta(self):
        return (self.loraB @ self.loraA) * self.scale


def applyLora(backbone, rank, alpha, targets):
    """Wrap every nn.Linear of a LeVJEPA encoder's blocks whose name contains one of `targets` (qkv, out, mlp)."""
    wrapped = []
    for name, module in list(backbone.named_modules()):
        for childName, child in list(module.named_children()):
            full = f"{name}.{childName}" if name else childName
            if isinstance(child, nn.Linear) and full.startswith("blocks.") and any(t in full for t in targets):
                lora = LoRALinear(child, rank, alpha).to(child.weight.device)
                setattr(module, childName, lora)
                wrapped.append(full)
    assert wrapped, f"no linear layer matched LoRA targets {targets}"
    return wrapped


def loraParameters(backbone):
    return [p for n, p in backbone.named_parameters() if n.endswith(("loraA", "loraB"))]


def mergedStateDict(backbone):
    """The state dict of the plain encoder: each LoRA layer's update folded into its base weight, so the saved
    checkpoint loads into the original architecture (embed.py, probe.py, the site adapter)."""
    deltas = {n: m.delta() for n, m in backbone.named_modules() if isinstance(m, LoRALinear)}
    out = {}
    for key, value in backbone.state_dict().items():
        if key.endswith(("loraA", "loraB")):
            continue
        parts = key.split(".")
        if len(parts) >= 3 and parts[-2] == "base" and ".".join(parts[:-2]) in deltas:
            layer = ".".join(parts[:-2])
            value = value + deltas[layer] if parts[-1] == "weight" else value
            key = f"{layer}.{parts[-1]}"
        out[key] = value.detach().clone()
    return out


def removeLora(backbone):
    """Put the plain nn.Linear layers back (adapters discarded); load a merged state dict afterwards."""
    for name, module in list(backbone.named_modules()):
        for childName, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, childName, child.base)


def loadTrainedTagger(run):
    """A finished finetuneTags.py run directory (checkpoint.pt + config.json, head.pt, vocab.json) -> (FontTagger,
    vocab), with either backbone kind."""
    backbone, kind, _ = loadBackbone(run)
    with open(os.path.join(run, "vocab.json")) as f:
        vocab = json.load(f)
    head = torch.load(os.path.join(run, "head.pt"), map_location="cpu")
    model = FontTagger(backbone, len(vocab), hidden=head["2.weight"].shape[0], kind=kind)
    model.head.load_state_dict(head)
    return model, vocab


def asymmetricLoss(logits, targets, gammaNeg=4.0, gammaPos=0.0, clip=0.05):
    """Asymmetric Loss (Ridnik et al., ICCV 2021), multi-label form."""
    p = torch.sigmoid(logits)
    pNeg = (p - clip).clamp(min=0)
    lossPos = targets * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gammaPos
    lossNeg = (1 - targets) * torch.log((1 - pNeg).clamp(min=1e-8)) * pNeg ** gammaNeg
    return -(lossPos + lossNeg).sum(dim=1).mean()


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def predict(model, glyphs, batchFonts=32, autocast=True):
    model.eval()
    out = []
    for start in range(0, len(glyphs), batchFonts):
        batch = torch.tensor(glyphs[start:start + batchFonts], device=device).float() / 255.0
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda" and autocast):
            out.append(model(batch).float().cpu())
    return torch.cat(out).numpy()


def top50Metrics(logits, targets, top50Idx):
    roc, pr = [], []
    for j in top50Idx:
        y = targets[:, j]
        if 0 < y.sum() < len(y):
            roc.append(roc_auc_score(y, logits[:, j]))
            pr.append(average_precision_score(y, logits[:, j]))
    return float(np.mean(roc)), float(np.mean(pr))


def ndcg(scores, rel):
    order = np.argsort(-scores)
    discounts = 1 / np.log2(np.arange(2, len(rel) + 2))
    ideal = np.sort(rel)[::-1]
    return float((rel[order] * discounts).sum() / (ideal * discounts).sum())


def iccvMetrics(logits, testTags, queries, vocabIndex):
    logProb = F.logsigmoid(torch.tensor(logits)).numpy()
    aps, ndcgs, skipped = [], [], 0
    for query in queries:
        if not all(t in vocabIndex for t in query):
            skipped += 1
            continue
        rel = np.array([all(t in tags for t in query) for tags in testTags], dtype=float)
        if rel.sum() == 0:
            continue
        scores = logProb[:, [vocabIndex[t] for t in query]].sum(axis=1)
        aps.append(average_precision_score(rel, scores))
        ndcgs.append(ndcg(scores, rel))
    return {"mAP": 100 * float(np.mean(aps)), "NDCG": 100 * float(np.mean(ndcgs)),
            "queries": len(aps), "skippedOutOfVocab": skipped}


def amtAccuracy(logits, testNames, vocabIndex, datasetDir):
    rows = list(csv.reader(open(os.path.join(datasetDir, "AMT-testset", "data.csv"))))[1:]
    index = {n: i for i, n in enumerate(testNames)}
    correct = total = 0
    for row in rows:
        tag, fonts, label = row[0].strip(), [f.strip() for f in row[1:4]], int(row[4])
        if tag not in vocabIndex or not all(f in index for f in fonts):
            continue
        scores = [logits[index[f], vocabIndex[tag]] for f in fonts]
        correct += int(np.argmax(scores) == label)
        total += 1
    return {"accuracy": correct / max(total, 1), "groups": total}


def testReport(model, split, vocab, vocabIndex, top50Idx, args):
    logits = predict(model, split["glyphs"], autocast=args.autocast)
    roc, pr = top50Metrics(logits, split["targets"], top50Idx)
    testTags = [split["tags"][n] for n in split["names"]]
    single = [[t] for t in readList(os.path.join(args.datasetDir, "myfonts-testset", "singletag-test"))]
    multi = [q.split("&&&") for q in readList(os.path.join(args.datasetDir, "myfonts-testset", "multitag-test"))]
    top300 = set(vocab[:300])  # vocab is sorted by train frequency
    return {
        "top50RocAuc": roc,
        "top50PrAuc": pr,
        "iccvSingle300": iccvMetrics(logits, testTags, [q for q in single if q[0] in top300], vocabIndex),
        "iccvSingleFull": iccvMetrics(logits, testTags, single, vocabIndex),
        "iccvMulti": iccvMetrics(logits, testTags, multi, vocabIndex),
        "amt": amtAccuracy(logits, split["names"], vocabIndex, args.datasetDir),
    }


def printReport(label, report):
    print(f"  [{label}] top-50 ROC-AUC {report['top50RocAuc']:.4f}  PR-AUC {report['top50PrAuc']:.4f} | "
          f"ICCV mAP single-300 {report['iccvSingle300']['mAP']:.2f}  single-full {report['iccvSingleFull']['mAP']:.2f}  "
          f"multi {report['iccvMulti']['mAP']:.2f} | AMT {report['amt']['accuracy']:.3f} (n={report['amt']['groups']})")


# ---------------------------------------------------------------- training

def main():
    args = parseArgs()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splitNames = {s: readList(os.path.join(args.datasetDir, "fontset", f"{s}set")) for s in ("train", "val", "test")}
    if args.maxFonts is not None:
        rng = random.Random(args.seed)
        splitNames = {s: sorted(rng.sample(n, min(args.maxFonts, len(n)))) for s, n in splitNames.items()}

    allNames = sorted(set().union(*splitNames.values()))
    cachedNames, cachedGlyphs = loadGlyphCache(args.datasetDir, allNames, args.fontSize, args.workers)
    glyphIndex = {n: i for i, n in enumerate(cachedNames)}
    allTags = loadTags(args.datasetDir, cachedNames)

    counts = Counter(t for n in splitNames["train"] if n in glyphIndex for t in allTags[n])
    vocab = [t for t, c in counts.most_common() if c >= args.minTagCount]
    vocabIndex = {t: i for i, t in enumerate(vocab)}
    top50Idx = list(range(min(50, len(vocab))))

    def buildSplit(names):
        names = [n for n in names if n in glyphIndex]
        targets = np.zeros((len(names), len(vocab)), dtype=np.float32)
        for i, n in enumerate(names):
            for t in allTags[n]:
                if t in vocabIndex:
                    targets[i, vocabIndex[t]] = 1
        return {"names": names, "glyphs": cachedGlyphs[[glyphIndex[n] for n in names]],
                "targets": targets, "tags": {n: allTags[n] for n in names}}

    splits = {s: buildSplit(n) for s, n in splitNames.items()}
    print(f"fonts train/val/test = {len(splits['train']['names'])}/{len(splits['val']['names'])}/"
          f"{len(splits['test']['names'])}, vocab = {len(vocab)} tags")

    backbone, backboneKind, saveBackboneConfig = loadBackbone(args.backbone)
    print(f"backbone: {args.backbone} ({backboneKind})")
    model = FontTagger(backbone, len(vocab), args.hidden, kind=backboneKind).to(device)

    lossFn = asymmetricLoss if args.loss == "asl" else \
        (lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets))

    stamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    outPath = args.evalOnly or os.path.join(args.outDir, stamp)
    os.makedirs(outPath, exist_ok=True)

    run = None
    if args.wandb and not args.evalOnly:
        import wandb
        run = wandb.init(entity="dylanberndt123-missouri-state-university", project="Font Retrieval",
                         config=vars(args), name=f"finetuneTags {stamp}")

    train = splits["train"]
    trainTargets = torch.tensor(train["targets"])
    stepsPerEpoch = max(1, len(train["names"]) // args.batchFonts)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and args.autocast)

    loraRank = getattr(args, "loraRank", 0)   # 0 = full finetuning; > 0 = LoRA adapters on a frozen LeVJEPA encoder
    if loraRank:
        assert backboneKind == "levjepa", "loraRank is implemented for LeVJEPA backbones only"

    def makeOptimizer(finetune):
        backbone.requires_grad_(finetune and not loraRank)
        groups = [{"params": model.head.parameters(), "lr": args.headLR}]
        if finetune and loraRank:
            wrapped = applyLora(backbone, loraRank, getattr(args, "loraAlpha", 2 * loraRank),
                                getattr(args, "loraTargets", ["qkv", "out", "mlp"]))
            backbone.requires_grad_(False)
            adapters = loraParameters(backbone)
            for p in adapters:
                p.requires_grad_(True)
            print(f"LoRA rank {loraRank}: {len(wrapped)} layers, {sum(p.numel() for p in adapters) / 1e6:.2f}M "
                  f"trainable backbone parameters of {sum(p.numel() for p in backbone.parameters()) / 1e6:.2f}M",
                  flush=True)
            groups.append({"params": adapters, "lr": args.loraLR})
        elif finetune:
            groups.append({"params": backbone.parameters(), "lr": args.backboneLR})
        opt = torch.optim.AdamW(groups, weight_decay=args.weightDecay)
        epochs = args.headOnlyEpochs if not finetune else max(1, args.epochs - args.headOnlyEpochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * stepsPerEpoch, pct_start=0.1)
        return opt, sched

    optimizer, scheduler = makeOptimizer(finetune=False)
    best = {"valRocAuc": -1, "epoch": None}
    frozenBaseline = None
    badEpochs = 0

    for epoch in range(0 if args.evalOnly else min(args.epochs, args.maxEpochs or args.epochs)):
        if epoch == args.headOnlyEpochs:
            # end of head-only phase == frozen-backbone baseline, measured with this exact code
            frozenBaseline = testReport(model, splits["test"], vocab, vocabIndex, top50Idx, args)
            printReport("frozen baseline, test", frozenBaseline)
            optimizer, scheduler = makeOptimizer(finetune=True)
            best["valRocAuc"] = -1  # early stopping restarts for the finetuning phase

        model.train()
        backbone.train(epoch >= args.headOnlyEpochs)
        order = np.random.permutation(len(train["names"]))
        totalLoss = 0.0
        for step in range(stepsPerEpoch):
            idx = order[step * args.batchFonts:(step + 1) * args.batchFonts]
            letters = np.stack([np.random.choice(26, args.glyphsPerFont, replace=False) for _ in idx])
            batch = torch.tensor(train["glyphs"][idx[:, None], letters], device=device).float() / 255.0
            if args.shiftAug > 0:
                dx, dy = np.random.randint(-args.shiftAug, args.shiftAug + 1, size=2)
                batch = torch.roll(batch, shifts=(int(dy), int(dx)), dims=(2, 3))
            targets = trainTargets[idx].to(device)

            with torch.autocast(device_type=device, enabled=device == "cuda" and args.autocast):
                logits = model(batch, torch.from_numpy(letters).to(device))
            loss = lossFn(logits.float(), targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            totalLoss += loss.item()

        valLogits = predict(model, splits["val"]["glyphs"], autocast=args.autocast)
        valRoc, valPr = top50Metrics(valLogits, splits["val"]["targets"], top50Idx)
        phase = "head-only" if epoch < args.headOnlyEpochs else "finetune"
        print(f"epoch {epoch + 1:>3} [{phase}] loss {totalLoss / stepsPerEpoch:.4f}  "
              f"val top-50 ROC-AUC {valRoc:.4f}  PR-AUC {valPr:.4f}")
        if run is not None:
            run.log({"loss": totalLoss / stepsPerEpoch, "val top50 ROC-AUC": valRoc,
                     "val top50 PR-AUC": valPr, "epoch": epoch + 1})

        if valRoc > best["valRocAuc"]:
            best = {"valRocAuc": valRoc, "valPrAuc": valPr, "epoch": epoch + 1, "phase": phase}
            badEpochs = 0
            torch.save(mergedStateDict(backbone) if loraRank and epoch >= args.headOnlyEpochs else backbone.state_dict(),
                       os.path.join(outPath, "checkpoint.pt"))
            torch.save(model.head.state_dict(), os.path.join(outPath, "head.pt"))
        elif epoch >= args.headOnlyEpochs:
            badEpochs += 1
            if badEpochs >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break

    if frozenBaseline is None and not args.evalOnly:  # run ended inside the head-only phase
        frozenBaseline = testReport(model, splits["test"], vocab, vocabIndex, top50Idx, args)

    # reload the best finetuned weights (checkpoint.pt loads with ViT.load(outPath)); a LoRA run saved them merged
    # into the plain layers, so the adapters come off first
    removeLora(backbone)
    backbone.load_state_dict(torch.load(os.path.join(outPath, "checkpoint.pt"), map_location=device))
    model.head.load_state_dict(torch.load(os.path.join(outPath, "head.pt"), map_location=device))
    finetuned = testReport(model, splits["test"], vocab, vocabIndex, top50Idx, args)

    if args.evalOnly:
        valRoc, valPr = top50Metrics(predict(model, splits["val"]["glyphs"], autocast=args.autocast),
                                     splits["val"]["targets"], top50Idx)
        best = {"valRocAuc": valRoc, "valPrAuc": valPr, "epoch": None, "phase": "finetune"}
    print(f"\nbest epoch {best['epoch']} ({best['phase']}), val top-50 ROC-AUC {best['valRocAuc']:.4f}")
    if frozenBaseline is not None:
        printReport("frozen baseline, test", frozenBaseline)
    printReport("finetuned, test", finetuned)

    saveBackboneConfig(outPath)
    with open(os.path.join(outPath, "runConfig.json"), "w") as f:
        json.dump(args.sections, f, indent=2)   # the configs/finetuneTags.json this run used
    with open(os.path.join(outPath, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(outPath, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "best": best, "frozenBaseline": frozenBaseline, "finetuned": finetuned}, f, indent=2)
    if run is not None:
        run.summary.update({"test top50 ROC-AUC": finetuned["top50RocAuc"], "test top50 PR-AUC": finetuned["top50PrAuc"],
                            "frozen test top50 ROC-AUC": frozenBaseline["top50RocAuc"]})
        run.finish()
    print(f"saved to {outPath}")


if __name__ == "__main__":
    main()
