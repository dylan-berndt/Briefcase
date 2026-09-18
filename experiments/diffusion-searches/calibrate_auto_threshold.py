"""
Properly calibrates a per-depth auto-descend threshold via an ROC-style
sweep, instead of guessing round numbers. For each tree depth: on
held-out examples, sweep confidence thresholds and compute (a)
PRECISION = P(top-1 correct | top-1 confidence >= threshold) -- the
false-auto-descend rate is 1-precision, and (b) COVERAGE = fraction of
examples at that depth whose confidence clears the threshold (how
often we'd actually get to skip asking). Picks, per depth, the LOWEST
threshold that hits a target precision -- this maximizes how often we
can safely auto-descend while keeping the false-descend rate at or
below the target, rather than guessing.

A false auto-descend is worse than typical oracle noise: the oracle
still picks from a state-verified-plausible candidate set, but a wrong
auto-descend silently discards the whole subtree with zero correction
chance. Default target precision is high (0.98) for that reason.

    python3 experiments/diffusion-searches/calibrate_auto_threshold.py \
        --classifierDir checkpoints/nav_classifier_tree6 --maxDepth 6
"""
import argparse
import json
import os

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, loadRaw, splitQueryCache, EMBEDDINGS_PATH
from train_navigation_classifier import NavigationClassifier, buildExamples, NavDataset
from torch.utils.data import DataLoader


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/nav_classifier_tree6")
    parser.add_argument("--treeCache", default="tree_variant_4_4_4_5_5_15.pkl")
    parser.add_argument("--sentenceModel", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--maxDepth", type=int, default=6)
    parser.add_argument("--maxTestExamples", type=int, default=40000)
    parser.add_argument("--targetPrecisions", default="0.95,0.98,0.99",
                         help="Comma-separated target precisions to calibrate a threshold for, per depth.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cpu"

    with open(os.path.join(args.classifierDir, "config.json")) as f:
        clsConfig = json.load(f)

    fontEmbeddings, sentenceCache = loadRaw(sentenceCachePath=cachePathFor(args.sentenceModel))
    names = sorted(fontEmbeddings.keys())
    baseConfigPath = os.path.join(clsConfig["baseCheckpoint"], "config.json")
    with open(baseConfigPath) as f:
        baseConfig = json.load(f)
    trainCache, testCache = splitQueryCache(sentenceCache, names, testFraction=baseConfig["testFraction"],
                                              seed=baseConfig["seed"])
    whitener = PCAWhitener.load(os.path.join(clsConfig["baseCheckpoint"], "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    hier = HierarchicalClusterIndex.load(args.treeCache)

    testPairs = [(name, i) for name, vectors in testCache.items() for i in range(len(vectors))]
    print(f"{len(testPairs)} held-out test pairs available")

    testExamples, depthCounts = buildExamples(testPairs, testCache, corpus, hier, args.maxDepth,
                                                args.maxTestExamples, args.seed + 1)
    print(f"{len(testExamples)} test examples (by depth: {depthCounts})")

    loader = DataLoader(NavDataset(testExamples), batch_size=512, shuffle=False, collate_fn=NavDataset.collate)

    model = NavigationClassifier(clsConfig["textDim"], clsConfig["pcaDim"], clsConfig["hiddenDim"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.classifierDir, "checkpoint.pt"), map_location=device))
    model.eval()

    confByDepth = {}
    correctByDepth = {}
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            node = batch["node"].to(device)
            children = batch["children"].to(device)
            mask = batch["mask"].to(device)
            trueIdx = batch["trueIdx"].to(device)
            depth = batch["depth"]

            scores = model(text, node, children, mask)
            probs = torch.softmax(scores, dim=1)
            conf, pred = probs.max(dim=1)
            correct = (pred == trueIdx)

            for d in depth.unique().tolist():
                m = (depth == d)
                confByDepth.setdefault(d, []).extend(conf[m].tolist())
                correctByDepth.setdefault(d, []).extend(correct[m].tolist())

    targetPrecisions = [float(p) for p in args.targetPrecisions.split(",")]
    calibrated = {p: {} for p in targetPrecisions}

    print("\nPer-depth threshold sweep (precision = P(correct | confidence>=t), coverage = fraction reaching t):")
    for d in sorted(confByDepth):
        conf = np.array(confByDepth[d])
        correct = np.array(correctByDepth[d])
        order = np.argsort(-conf)
        confSorted = conf[order]
        correctSorted = correct[order]
        cumCorrect = np.cumsum(correctSorted)
        cumCount = np.arange(1, len(correctSorted) + 1)
        precisionAtRank = cumCorrect / cumCount  # precision if threshold = confSorted[i] (top i+1 most confident)

        print(f"  depth {d} (n={len(conf)}, overall accuracy={correct.mean():.3f}):")
        for targetP in targetPrecisions:
            # find the largest prefix (highest coverage) whose precision still clears targetP
            okIdx = np.where(precisionAtRank >= targetP)[0]
            if len(okIdx) == 0:
                threshold, coverage, achieved = 1.01, 0.0, float("nan")
            else:
                bestI = okIdx.max()
                threshold = float(confSorted[bestI])
                coverage = (bestI + 1) / len(conf)
                achieved = float(precisionAtRank[bestI])
            calibrated[targetP][d] = threshold
            print(f"    target precision {targetP}: threshold={threshold:.4f}  "
                  f"coverage={coverage:.3f}  achieved precision={achieved:.4f}")

    print("\nCalibrated --autoThresholds schedules (pass directly to evaluate_classifier_branching.py):")
    maxDepthSeen = max(confByDepth.keys())
    for targetP in targetPrecisions:
        schedule = [calibrated[targetP].get(d, 1.01) for d in range(maxDepthSeen + 1)]
        print(f"  target={targetP}: {','.join(f'{t:.4f}' for t in schedule)}")


if __name__ == "__main__":
    main()
