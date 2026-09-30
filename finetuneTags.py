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
# Font representation: mean of the L2-normalized per-glyph CLS vectors, the same pooling
# utils/embeddings.generateEmbeddings uses for embeddings/all.json. Training samples
# --glyphsPerFont random letters per font per step; evaluation uses all 26.
#
# No rotation/aspect augmentation: tags describe slant and width, so those augmentations would
# randomize the very attributes being predicted. --shiftAug adds a small random translation only.
#
# Metrics (all on the official splits):
#   - top-50 macro ROC-AUC / PR-AUC, the music auto-tagging protocol (Won et al. 2020). Val
#     top-50 ROC-AUC is the early-stopping criterion.
#   - ICCV MyFonts-test protocol: single-tag (top-300 / full) and multi-tag mAP + NDCG.
#   - AMT-test accuracy (human "which of 3 fonts best fits this tag").
# Test metrics are reported twice from the same code: at the end of the head-only phase
# ("frozenBaseline") and for the best finetuned epoch.
#
#   python finetuneTags.py                            # full run (GPU)
#   python finetuneTags.py --maxFonts 300 --epochs 2  # smoke test

import argparse
import csv
import json
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

from utils.vit import ViT
from utils.loaders.myfonts import loadRochesterImage

LETTERS = "abcdefghijklmnopqrstuvwxyz"
device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--datasetDir", default="dataset")
    p.add_argument("--backbone", default=os.path.join("checkpoints", "pretrain", "best"))
    p.add_argument("--outDir", default=os.path.join("checkpoints", "retrieval", "finetuneTags"))
    p.add_argument("--fontSize", type=int, default=32)
    p.add_argument("--minTagCount", type=int, default=1, help="1 keeps every train tag, so the ICCV query sets stay comparable")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--headOnlyEpochs", type=int, default=3)
    p.add_argument("--batchFonts", type=int, default=64)
    p.add_argument("--glyphsPerFont", type=int, default=8)
    p.add_argument("--headLR", type=float, default=1e-3)
    p.add_argument("--backboneLR", type=float, default=1e-5)
    p.add_argument("--weightDecay", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--loss", choices=["bce", "asl"], default="bce")
    p.add_argument("--shiftAug", type=int, default=0, help="max random translation in pixels (0 = off)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--maxFonts", type=int, default=None, help="subsample fonts per split, for smoke tests")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


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


class FontTagger(nn.Module):
    def __init__(self, vit, numTags, hidden):
        super().__init__()
        self.vit = vit
        dim = vit.config.embedDim
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.2),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, numTags),
        )

    def fontFeatures(self, glyphs):
        """glyphs: [B, G, H, W] -> [B, D], mean of per-glyph L2-normalized CLS (all.json pooling)."""
        B, G, H, W = glyphs.shape
        cls = clsFeatures(self.vit, glyphs.reshape(B * G, H, W))
        return F.normalize(cls, dim=-1).reshape(B, G, -1).mean(dim=1)

    def forward(self, glyphs):
        return self.head(self.fontFeatures(glyphs))


def asymmetricLoss(logits, targets, gammaNeg=4.0, gammaPos=0.0, clip=0.05):
    """Asymmetric Loss (Ridnik et al., ICCV 2021), multi-label form."""
    p = torch.sigmoid(logits)
    pNeg = (p - clip).clamp(min=0)
    lossPos = targets * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gammaPos
    lossNeg = (1 - targets) * torch.log((1 - pNeg).clamp(min=1e-8)) * pNeg ** gammaNeg
    return -(lossPos + lossNeg).sum(dim=1).mean()


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def predict(model, glyphs, batchFonts=32):
    model.eval()
    out = []
    for start in range(0, len(glyphs), batchFonts):
        batch = torch.tensor(glyphs[start:start + batchFonts], device=device).float() / 255.0
        with torch.autocast(device_type=device, enabled=device == "cuda"):
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
    logits = predict(model, split["glyphs"])
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

    vit, vitConfig = ViT.load(args.backbone)
    model = FontTagger(vit, len(vocab), args.hidden).to(device)

    lossFn = asymmetricLoss if args.loss == "asl" else \
        (lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets))

    stamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    outPath = os.path.join(args.outDir, stamp)
    os.makedirs(outPath, exist_ok=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity="dylanberndt123-missouri-state-university", project="Font Retrieval",
                         config=vars(args), name=f"finetuneTags {stamp}")

    train = splits["train"]
    trainTargets = torch.tensor(train["targets"])
    stepsPerEpoch = max(1, len(train["names"]) // args.batchFonts)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def makeOptimizer(finetune):
        vit.requires_grad_(finetune)
        groups = [{"params": model.head.parameters(), "lr": args.headLR}]
        if finetune:
            groups.append({"params": vit.parameters(), "lr": args.backboneLR})
        opt = torch.optim.AdamW(groups, weight_decay=args.weightDecay)
        epochs = args.headOnlyEpochs if not finetune else max(1, args.epochs - args.headOnlyEpochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * stepsPerEpoch, pct_start=0.1)
        return opt, sched

    optimizer, scheduler = makeOptimizer(finetune=False)
    best = {"valRocAuc": -1, "epoch": None}
    frozenBaseline = None
    badEpochs = 0

    for epoch in range(args.epochs):
        if epoch == args.headOnlyEpochs:
            # end of head-only phase == frozen-backbone baseline, measured with this exact code
            frozenBaseline = testReport(model, splits["test"], vocab, vocabIndex, top50Idx, args)
            printReport("frozen baseline, test", frozenBaseline)
            optimizer, scheduler = makeOptimizer(finetune=True)
            best["valRocAuc"] = -1  # early stopping restarts for the finetuning phase

        model.train()
        vit.train(epoch >= args.headOnlyEpochs)
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

            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(batch)
            loss = lossFn(logits.float(), targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            totalLoss += loss.item()

        valLogits = predict(model, splits["val"]["glyphs"])
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
            torch.save(vit.state_dict(), os.path.join(outPath, "checkpoint.pt"))
            torch.save(model.head.state_dict(), os.path.join(outPath, "head.pt"))
        elif epoch >= args.headOnlyEpochs:
            badEpochs += 1
            if badEpochs >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break

    if frozenBaseline is None:  # run ended inside the head-only phase
        frozenBaseline = testReport(model, splits["test"], vocab, vocabIndex, top50Idx, args)

    # reload the best finetuned weights (checkpoint.pt loads with ViT.load(outPath))
    vit.load_state_dict(torch.load(os.path.join(outPath, "checkpoint.pt"), map_location=device))
    model.head.load_state_dict(torch.load(os.path.join(outPath, "head.pt"), map_location=device))
    finetuned = testReport(model, splits["test"], vocab, vocabIndex, top50Idx, args)

    print(f"\nbest epoch {best['epoch']} ({best['phase']}), val top-50 ROC-AUC {best['valRocAuc']:.4f}")
    printReport("frozen baseline, test", frozenBaseline)
    printReport("finetuned, test", finetuned)

    vitConfig.save(os.path.join(outPath, "config.json"))
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
