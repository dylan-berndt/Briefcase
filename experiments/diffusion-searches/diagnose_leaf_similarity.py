"""
Validates the "leaf-success rate" metric before trusting it: does
landing in the correct leaf actually mean landing near the true font,
or can a leaf contain fonts that aren't meaningfully similar to each
other?

CORRECTED: the first version of this script measured "fraction of
leaf-mates within a member's own top-10 nearest neighbors," which has
a hidden ceiling effect the user caught directly -- with a FIXED
top-10 neighbor set and a leaf of size N, that fraction can never
exceed min(1, 10/(N-1)) regardless of actual similarity, so large
leaves were mechanically penalized by their own size, not measured for
coherence. This version instead computes direct pairwise cosine
similarity among leaf members (no size-dependent ceiling) and compares
it to same-size random-group baselines.

    python3 experiments/diffusion-searches/diagnose_leaf_similarity.py \
        --treeCache tree_variant_4_4_4_5_5_15.pkl
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treeCache", default="tree_variant_15_15_5_5_15.pkl")
    parser.add_argument("--baseCheckpoint", default="checkpoints/nav_classifier_v4")
    parser.add_argument("--maxLeaves", type=int, default=1000, help="Sample this many leaves (cheap, no need for all).")
    parser.add_argument("--randomTrials", type=int, default=50, help="Random same-size groups per bucket, for the baseline.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def collectLeaves(node, out):
    if not node.children:
        out.append(node)
    else:
        for c in node.children:
            collectLeaves(c, out)


def leafBucket(size):
    if size <= 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 20:
        return "6-20"
    if size <= 50:
        return "21-50"
    return "51+"


def meanPairwiseCosine(vecs):
    n = vecs.shape[0]
    if n <= 1:
        return 1.0
    normed = F.normalize(vecs, dim=1)
    sim = normed @ normed.T
    offDiagSum = sim.sum().item() - n  # diagonal is all 1.0
    return offDiagSum / (n * (n - 1))


def main():
    args = parseArgs()
    device = "cpu"

    with open(f"{args.baseCheckpoint}/config.json") as f:
        clsConfig = json.load(f)
    whitener = PCAWhitener.load(f"{clsConfig['baseCheckpoint']}/whitener.npz")
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    hier = HierarchicalClusterIndex.load(args.treeCache)

    leaves = []
    collectLeaves(hier.root, leaves)
    print(f"{len(leaves)} total leaves")

    rng = np.random.RandomState(args.seed)
    sample = leaves if len(leaves) <= args.maxLeaves else [leaves[i] for i in
                                                              rng.choice(len(leaves), args.maxLeaves, replace=False)]

    buckets = {"1": [], "2-5": [], "6-20": [], "21-50": [], "51+": []}
    bucketSizes = {"1": [], "2-5": [], "6-20": [], "21-50": [], "51+": []}
    for leaf in sample:
        members = leaf.memberIndices
        size = len(members)
        vecs = corpus.whitenedMatrix[members]
        sim = meanPairwiseCosine(vecs)
        bucket = leafBucket(size)
        buckets[bucket].append(sim)
        bucketSizes[bucket].append(size)

    randomBaselines = {}
    for bucket, sizes in bucketSizes.items():
        if not sizes:
            continue
        repSize = int(np.median(sizes))
        repSize = max(repSize, 2)
        trials = []
        for _ in range(args.randomTrials):
            idx = rng.choice(len(corpus.names), repSize, replace=False)
            trials.append(meanPairwiseCosine(corpus.whitenedMatrix[idx]))
        randomBaselines[bucket] = float(np.mean(trials))

    print("\nMean intra-leaf pairwise cosine similarity (whitened space), by leaf-size bucket "
          "(no size-dependent ceiling, unlike the earlier top-10-membership version):")
    for bucket in ["1", "2-5", "6-20", "21-50", "51+"]:
        vals = buckets[bucket]
        if vals:
            base = randomBaselines.get(bucket, float("nan"))
            print(f"  leaf size {bucket}: n={len(vals)} leaves, mean intra-leaf cosine={np.mean(vals):.4f}  "
                  f"(random same-size-group baseline: {base:.4f})")


if __name__ == "__main__":
    main()
