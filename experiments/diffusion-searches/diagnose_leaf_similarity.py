"""
Validates the "leaf-success rate" metric before trusting it: does
landing in the correct leaf actually mean landing near the true font,
or can a leaf contain fonts that aren't meaningfully similar to each
other? If leaf-mates aren't typically each other's near neighbors,
leaf-success doesn't measure anything useful.

Uses the SAME notion of similarity as the acceptance-set recall@k
metric elsewhere in this investigation (top-10 nearest neighbors in
the whitened space), not a new ad-hoc one: for each leaf member, what
fraction of its OTHER leaf-mates are within ITS OWN top-K nearest
neighbors in the full corpus? If leaf-mates are genuinely similar,
this should be high; if the leaf is just a k-means bucket with no
real coherence, it'll be low.

Breaks results out by leaf-size bucket, since the tree's leaf sizes
are bimodal (median 3, but a few outliers up to 150 -- a known
k-means degenerate-cluster artifact from near-duplicate embeddings) --
the concern is specifically whether those large leaves are genuinely
tight clusters or a "junk drawer" that would make leaf-success
misleading.

    python3 experiments/diffusion-searches/diagnose_leaf_similarity.py \
        --treeCache tree_variant_4_4_4_5_5_15.pkl
"""
import argparse

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treeCache", default="tree_variant_15_15_5_5_15.pkl")
    parser.add_argument("--baseCheckpoint", default="checkpoints/nav_classifier_v4")
    parser.add_argument("--acceptanceSize", type=int, default=10,
                         help="Same K as the recall@k acceptance set elsewhere in this investigation.")
    parser.add_argument("--maxLeaves", type=int, default=500,
                         help="Sample this many leaves (cheap, no need for all; kept small to bound the "
                              "nearest-neighbor distance matrix's memory footprint).")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def collectLeaves(node, out):
    if not node.children:
        out.append(node)
    else:
        for c in node.children:
            collectLeaves(c, out)


def main():
    args = parseArgs()
    device = "cpu"

    import json
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

    K = args.acceptanceSize

    # Only compute nearest-neighbor sets for fonts that actually appear as members of a sampled leaf --
    # avoids materializing a full N x N distance matrix (39,421^2 floats would be several GB).
    neededIdx = sorted({m for leaf in sample for m in leaf.memberIndices.tolist()})
    neededVecs = corpus.whitenedMatrix[neededIdx]
    _, neighborRows = corpus.nearest(neededVecs, k=K + 1)  # +1 since a member's own vector is its own nearest neighbor
    neighborSets = {}
    for pos, idx in enumerate(neededIdx):
        names = [n for n in neighborRows[pos] if n != corpus.names[idx]][:K]
        neighborSets[idx] = {corpus.nameToIndex[n] for n in names}

    buckets = {"1": [], "2-5": [], "6-20": [], "21-50": [], "51+": []}

    def bucketFor(size):
        if size <= 1:
            return "1"
        if size <= 5:
            return "2-5"
        if size <= 20:
            return "6-20"
        if size <= 50:
            return "21-50"
        return "51+"

    randomBaselineHits = []
    for leaf in sample:
        members = leaf.memberIndices.tolist()
        size = len(members)
        if size <= 1:
            buckets[bucketFor(size)].append(1.0)  # trivially "similar to itself"
            continue
        hitFracs = []
        for m in members:
            others = [o for o in members if o != m]
            hits = sum(1 for o in others if o in neighborSets[m])
            hitFracs.append(hits / len(others))
        buckets[bucketFor(size)].append(float(np.mean(hitFracs)))

        # random-pairs baseline: same member count, random OTHER corpus indices instead of real leaf-mates
        randomOthers = rng.choice(len(corpus.names), size - 1, replace=False)
        m0 = members[0]
        randHits = sum(1 for o in randomOthers if o in neighborSets[m0])
        randomBaselineHits.append(randHits / (size - 1))

    print(f"\nFraction of leaf-mates that are ALSO within a member's own top-{K} nearest neighbors "
          f"(1.0 = leaf-mates are essentially always 'acceptance-set-equivalent' close; "
          f"random-pairs baseline ~{np.mean(randomBaselineHits) if randomBaselineHits else 0:.4f}):")
    for bucket, vals in buckets.items():
        if vals:
            print(f"  leaf size {bucket}: n={len(vals)} leaves, mean fraction={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
