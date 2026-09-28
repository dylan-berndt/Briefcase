"""
General-purpose tree-variant builder: same branching schedule as the
original results/trees/tree_variant_15_15_5_5_15.pkl, but with a chosen metric
("euclidean"/"cosine") and split algorithm ("kmeans"/"bisecting") --
see corpus.HierarchicalClusterIndex.fit's docstring for what each does
and why. Prints root/depth-1 size-balance stats so different variants
can be compared directly against the original tree without retraining
a classifier first.

    python3 experiments/diffusion-searches/tree_construction/build_tree_variant.py \
        --metric euclidean --splitAlgorithm bisecting \
        --outPath results/trees/tree_variant_bisecting_15_15_5_5_15.pkl
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_baseDir = _os.path.dirname(_scriptDir)
for _p in [_baseDir] + [_os.path.join(_baseDir, _d) for _d in _os.listdir(_baseDir)
                         if _os.path.isdir(_os.path.join(_baseDir, _d))]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import json
import time

import numpy as np

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH

BASE_CHECKPOINT = "checkpoints/diffusion-searches/navigation_classifier/scale/nav_classifier_v4"
BRANCHING = [15, 15, 5, 5, 15]
MAX_DEPTH = 5
MIN_LEAF_SIZE = 5


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--splitAlgorithm", choices=["kmeans", "bisecting"], default="kmeans")
    parser.add_argument("--outPath", required=True)
    return parser.parse_args()


def sizeStats(root):
    depthSizes = {}
    leafSizes = []

    def walk(node, depth):
        if not node.children:
            leafSizes.append(len(node.memberIndices))
            return
        depthSizes.setdefault(depth, []).append([len(c.memberIndices) for c in node.children])
        for c in node.children:
            walk(c, depth + 1)

    walk(root, 0)
    return depthSizes, np.array(leafSizes)


def main():
    args = parseArgs()
    with open(f"{BASE_CHECKPOINT}/config.json") as f:
        clsConfig = json.load(f)
    whitener = PCAWhitener.load(f"{clsConfig['baseCheckpoint']}/whitener.npz")
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device="cpu")

    print(f"building tree: metric={args.metric} splitAlgorithm={args.splitAlgorithm} "
          f"branching={BRANCHING} maxDepth={MAX_DEPTH} minLeafSize={MIN_LEAF_SIZE} ...")
    start = time.time()
    tree = HierarchicalClusterIndex.fit(corpus, branchingFactor=BRANCHING, maxDepth=MAX_DEPTH,
                                         minLeafSize=MIN_LEAF_SIZE, metric=args.metric,
                                         splitAlgorithm=args.splitAlgorithm)
    print(f"built in {time.time() - start:.1f}s")
    tree.save(args.outPath)
    print(f"saved to {args.outPath}")

    depthSizes, leafSizes = sizeStats(tree.root)
    for depth in sorted(depthSizes):
        allSizes = [s for group in depthSizes[depth] for s in group]
        groupMaxImbalance = max(max(group) / (sum(group) / len(group)) for group in depthSizes[depth])
        print(f"  depth {depth}: n_children_total={len(allSizes)} size mean={np.mean(allSizes):.0f} "
              f"max={max(allSizes)} min={min(allSizes)} worst single-node (max/mean) imbalance={groupMaxImbalance:.2f}x")
    print(f"  leaves: {len(leafSizes)} mean={leafSizes.mean():.1f} median={np.median(leafSizes):.0f} "
          f"min={leafSizes.min()} max={leafSizes.max()}")
    print(f"\nroot (depth 0) child sizes: {sorted([len(c.memberIndices) for c in tree.root.children], reverse=True)}")


if __name__ == "__main__":
    main()
