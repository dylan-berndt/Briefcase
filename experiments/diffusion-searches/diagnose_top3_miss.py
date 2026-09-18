"""
Diagnoses the ~33% recall@100 ceiling noticed in the beam-search
results: for each held-out query, walk the TRUE tree path (ground
truth, via train_navigation_classifier.trueChildAt) and check, at
every level, whether the true child is inside the classifier's own
top-3 ranked children. If it's missed at ANY level, that session is
structurally unrecoverable by the current search protocol (hard-commit
or beam) -- both only ever consider the classifier's top-3 shown
candidates, so a miss there forecloses the true leaf regardless of
oracle noise or beam width.

    python3 experiments/diffusion-searches/diagnose_top3_miss.py \
        --classifierDir checkpoints/nav_classifier_v4 --maxQueries 1000
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH
from train_navigation_classifier import NavigationClassifier, trueChildAt
from evaluate_classifier_branching import scoreChildren, cachePathFor


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/nav_classifier_v4")
    parser.add_argument("--maxQueries", type=int, default=1000)
    parser.add_argument("--maxOptions", type=int, default=3)
    parser.add_argument("--maxDepth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9999)
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cpu"

    with open(os.path.join(args.classifierDir, "config.json")) as f:
        clsConfig = json.load(f)

    whitener = PCAWhitener.load(os.path.join(clsConfig["baseCheckpoint"], "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    hier = HierarchicalClusterIndex.load(clsConfig["treeCache"])

    with open(cachePathFor(clsConfig["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)
    with open(os.path.join(clsConfig["baseCheckpoint"], "test_pairs.json")) as f:
        testPairs = json.load(f)
    testPairs = [p for p in testPairs if p["font"] in corpus.nameToIndex]
    byFont = {}
    for p in testPairs:
        byFont.setdefault(p["font"], p)
    queries = list(byFont.values())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(queries)
    queries = queries[:args.maxQueries]
    print(f"{len(queries)} queries")

    model = NavigationClassifier(clsConfig["textDim"], clsConfig["pcaDim"], clsConfig["hiddenDim"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.classifierDir, "checkpoint.pt"), map_location=device))
    model.eval()

    firstMissDepth = []  # None if never missed along the true path
    missByDepth = {d: 0 for d in range(args.maxDepth)}
    totalByDepth = {d: 0 for d in range(args.maxDepth)}
    everMissed = 0

    for qi, pair in enumerate(queries):
        text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32))
        targetIdx = corpus.nameToIndex[pair["font"]]

        node = hier.root
        depth = 0
        missedAt = None
        while node.children and depth < args.maxDepth:
            trueIdx, trueChild = trueChildAt(node, targetIdx)
            if trueIdx is None:
                break
            probs = scoreChildren(model, text, node, device)
            k = min(args.maxOptions, len(node.children))
            top3 = set(np.argsort(-probs)[:k].tolist())
            totalByDepth[depth] += 1
            if trueIdx not in top3:
                missByDepth[depth] += 1
                if missedAt is None:
                    missedAt = depth
            node = trueChild
            depth += 1

        firstMissDepth.append(missedAt)
        if missedAt is not None:
            everMissed += 1

        if (qi + 1) % 200 == 0 or qi + 1 == len(queries):
            print(f"{qi + 1}/{len(queries)}")

    total = len(queries)
    print(f"\nsessions where true child fell outside classifier's top-3 at SOME level: "
          f"{everMissed}/{total} = {everMissed/total:.4f}")
    print("miss rate by depth (of sessions that reached that depth still on the true path):")
    for d in range(args.maxDepth):
        if totalByDepth[d] > 0:
            print(f"  depth {d}: {missByDepth[d]}/{totalByDepth[d]} = {missByDepth[d]/totalByDepth[d]:.4f}")
    print("\nfirst-miss depth distribution (None = never missed):")
    from collections import Counter
    counts = Counter(firstMissDepth)
    for d in sorted([k for k in counts if k is not None]):
        print(f"  first miss at depth {d}: {counts[d]}")
    print(f"  never missed: {counts[None]}")


if __name__ == "__main__":
    main()
