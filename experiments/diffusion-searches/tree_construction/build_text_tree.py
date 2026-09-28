"""
Builds a text-informed tree variant: same branching schedule as
results/trees/tree_variant_15_15_5_5_15.pkl, but cluster ASSIGNMENT is done on
[visual whitened embedding; auxWeight * per-font mean text embedding]
instead of visual embedding alone -- see corpus.HierarchicalClusterIndex.
fit's auxMatrix/auxWeight docstring for the mechanism and motivation
(diagnose_split_coherence.py found a visually-adjacent 8,390-font
mega-cluster whose members have genuinely separable text descriptions --
neither the cosine-metric fix nor bisecting-balance fix addressed that,
both just relocated the resulting depth-1 confusion instead of removing
it, per the build_cosine_tree.py / build_tree_variant.py runs).

Per-font text signal: mean-pooled BGE embedding over that font's cached
queries (sentenceCache), looked up by DIRECT name match against
corpus.names (same convention diagnose_split_coherence.py's
representativeDescribed already relied on -- most corpus names that
have any text data are valid sentenceCache keys directly, no trie
matching needed). Fonts with no sentenceCache entry get an all-zero
text row, so their placement falls back to visual-only, same as before.

    python3 experiments/diffusion-searches/tree_construction/build_text_tree.py --auxWeight 1.0
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
import pickle
import time

import numpy as np
from pygtrie import CharTrie

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH, loadTfidfFeatureCache, splitQueryCache

BASE_CHECKPOINT = "checkpoints/diffusion-searches/navigation_classifier/scale/nav_classifier_v4"
BRANCHING = [15, 15, 5, 5, 15]
MAX_DEPTH = 5
MIN_LEAF_SIZE = 5
SENTENCE_CACHE_PATH = "embeddings/sentenceQueries_BAAI_bge-large-en-v1.5.pkl"


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textSource", choices=["bge", "lsa"], default="bge",
                         help="'bge': dense sentence-transformer embedding, mean-pooled per font. "
                              "'lsa': precomputed TF-IDF/TruncatedSVD features (build_tfidf_cache.py) -- "
                              "an explicitly linear, variance-preserving projection, so Euclidean/cosine "
                              "distance between per-font centroids is literally what it was built to "
                              "respect, unlike a contrastively-trained dense embedding.")
    parser.add_argument("--tfidfDim", type=int, default=256)
    parser.add_argument("--auxWeight", type=float, default=1.0)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="cosine",
                         help="visual-block normalization -- cosine puts it on equal per-vector-norm "
                              "footing with the (always L2-normalized) text block, so auxWeight is "
                              "interpretable as a real relative-influence knob.")
    parser.add_argument("--splitAlgorithm", choices=["kmeans", "bisecting"], default="kmeans")
    parser.add_argument("--outPath", default=None)
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
    outPath = args.outPath or _os.path.join("results", "trees",
                                             f"tree_variant_text_{args.textSource}_w{args.auxWeight:g}_"
                                             f"{args.metric}_{args.splitAlgorithm}.pkl")

    with open(f"{BASE_CHECKPOINT}/config.json") as f:
        clsConfig = json.load(f)
    whitener = PCAWhitener.load(f"{clsConfig['baseCheckpoint']}/whitener.npz")
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device="cpu")

    if args.textSource == "lsa":
        sentenceCache = loadTfidfFeatureCache(args.tfidfDim)
    else:
        with open(SENTENCE_CACHE_PATH, "rb") as f:
            sentenceCache = pickle.load(f)

    # Use ONLY the TRAIN half of each font's cached queries -- same testFraction/seed
    # train_navigation_classifier.py itself splits on (checkpoints/diffusion-searches/diffusion_sampling/diffusion_lr_5e-4/
    # config.json, the baseCheckpoint every nav_classifier_* config points at). Mean-
    # pooling the FULL cache (train+test queries) would leak test-split query phrasing
    # into the tree structure the classifier is then evaluated on navigating -- diluted
    # by per-font averaging, but real, and worth avoiding rather than discovering later.
    with open(f"{clsConfig['baseCheckpoint']}/config.json") as f:
        baseConfig = json.load(f)
    sentenceCache, _ = splitQueryCache(sentenceCache, list(sentenceCache.keys()),
                                        testFraction=baseConfig["testFraction"], seed=baseConfig["seed"])

    trie = CharTrie()
    for key in sentenceCache:
        trie[key] = key

    textDim = next(iter(sentenceCache.values())).shape[1]
    auxMatrix = np.zeros((len(corpus.names), textDim), dtype=np.float32)
    covered = 0
    for i, name in enumerate(corpus.names):
        vecs = sentenceCache.get(name)
        if vecs is None:
            match = trie.longest_prefix(name.strip())
            if match.key is not None:
                vecs = sentenceCache[match.key]
        if vecs is not None:
            auxMatrix[i] = np.asarray(vecs, dtype=np.float32).mean(axis=0)
            covered += 1
    print(f"text coverage: {covered}/{len(corpus.names)} corpus fonts matched to sentenceCache "
          f"(direct + longest-prefix trie match) ({covered / len(corpus.names):.1%})")

    print(f"building TEXT-INFORMED tree: metric={args.metric} splitAlgorithm={args.splitAlgorithm} "
          f"auxWeight={args.auxWeight} branching={BRANCHING} maxDepth={MAX_DEPTH} minLeafSize={MIN_LEAF_SIZE} ...")
    start = time.time()
    tree = HierarchicalClusterIndex.fit(corpus, branchingFactor=BRANCHING, maxDepth=MAX_DEPTH,
                                         minLeafSize=MIN_LEAF_SIZE, metric=args.metric,
                                         splitAlgorithm=args.splitAlgorithm,
                                         auxMatrix=auxMatrix, auxWeight=args.auxWeight)
    print(f"built in {time.time() - start:.1f}s")
    tree.save(outPath)
    print(f"saved to {outPath}")

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
