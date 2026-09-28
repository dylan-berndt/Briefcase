"""
Builds a cosine-split version of the existing results/trees/tree_variant_15_15_5_5_15.pkl
(identical branching schedule, maxDepth, minLeafSize, randomState) to
directly test the open Euclidean-vs-cosine question (diagnose_metric_
choice.py) at the level that actually matters: does it change the TREE
STRUCTURE in a way that fixes or reduces the mega-cluster imbalance found
by diagnose_split_coherence.py (root child 5, 8390/39421 members, ~21%
of the whole corpus in one of 15 root branches)? See
corpus.HierarchicalClusterIndex.fit's metric="cosine" docstring for how
this isolates the split itself from the larger, unstarted question of
switching the whole pipeline's distance metric.

    python3 experiments/diffusion-searches/tree_construction/build_cosine_tree.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_baseDir = _os.path.dirname(_scriptDir)
for _p in [_baseDir] + [_os.path.join(_baseDir, _d) for _d in _os.listdir(_baseDir)
                         if _os.path.isdir(_os.path.join(_baseDir, _d))]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import time

import numpy as np

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH

BASE_CHECKPOINT = "checkpoints/diffusion-searches/navigation_classifier/scale/nav_classifier_v4"
OUT_PATH = "results/trees/tree_variant_cosine_15_15_5_5_15.pkl"
BRANCHING = [15, 15, 5, 5, 15]
MAX_DEPTH = 5
MIN_LEAF_SIZE = 5


def sizeStats(root):
    sizes = {}
    leafSizes = []

    def walk(node, depth):
        if not node.children:
            leafSizes.append(len(node.memberIndices))
            return
        sizes.setdefault(depth, []).append([len(c.memberIndices) for c in node.children])
        for c in node.children:
            walk(c, depth + 1)

    walk(root, 0)
    return sizes, np.array(leafSizes)


def main():
    with open(f"{BASE_CHECKPOINT}/config.json") as f:
        clsConfig = json.load(f)
    whitener = PCAWhitener.load(f"{clsConfig['baseCheckpoint']}/whitener.npz")
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device="cpu")

    print(f"building COSINE tree: branching={BRANCHING}, maxDepth={MAX_DEPTH}, minLeafSize={MIN_LEAF_SIZE} ...")
    start = time.time()
    cosineTree = HierarchicalClusterIndex.fit(corpus, branchingFactor=BRANCHING, maxDepth=MAX_DEPTH,
                                               minLeafSize=MIN_LEAF_SIZE, metric="cosine")
    print(f"built in {time.time() - start:.1f}s")
    cosineTree.save(OUT_PATH)
    print(f"saved to {OUT_PATH}")

    euclideanTree = HierarchicalClusterIndex.load(clsConfig["treeCache"])

    for label, tree in [("EUCLIDEAN (existing)", euclideanTree), ("COSINE (new)", cosineTree)]:
        depthSizes, leafSizes = sizeStats(tree.root)
        print(f"\n=== {label} ===")
        for depth in sorted(depthSizes):
            allSizes = [s for group in depthSizes[depth] for s in group]
            groupMaxImbalance = max(max(group) / (sum(group) / len(group)) for group in depthSizes[depth])
            print(f"  depth {depth}: n_children_total={len(allSizes)} size mean={np.mean(allSizes):.0f} "
                  f"max={max(allSizes)} min={min(allSizes)} "
                  f"worst single-node (max/mean) imbalance={groupMaxImbalance:.2f}x")
        print(f"  leaves: {len(leafSizes)} mean={leafSizes.mean():.1f} median={np.median(leafSizes):.0f} "
              f"min={leafSizes.min()} max={leafSizes.max()}")

    root = cosineTree.root
    print(f"\nroot (depth 0) child sizes, COSINE tree: {sorted([len(c.memberIndices) for c in root.children], reverse=True)}")
    print(f"root (depth 0) child sizes, EUCLIDEAN tree: {sorted([len(c.memberIndices) for c in euclideanTree.root.children], reverse=True)}")


if __name__ == "__main__":
    main()
