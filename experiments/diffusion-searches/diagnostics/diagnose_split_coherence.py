"""
Diagnoses WHY depth-0/1 top-3 miss rates are high (diagnose_top3_miss.py:
18.1%/14.4%, vs 1.3-4.8% at depths 2-4), by surfacing the actual confused
child pairs for human inspection -- not just their miss rate. Every prior
fix (depth-weighting, more data/capacity, tree restructuring) treated
depth-0/1 difficulty as a training problem. This checks a different
hypothesis first: maybe some of these splits are just k-means drawing an
arbitrary line through a coherent style region (fixable by building a
navigability-aware tree), rather than the classifier failing to learn a
real, describable boundary (not fixable by any tree, since the ambiguity
would be in the language itself).

For the most-confused (trueChild, predictedTop1Child) pairs at depth 0 and
depth 1, prints each child's real font descriptions side by side, plus the
existing geometric `hardness` diagnostic (HierarchicalClusterIndex.
labelHardness) already computed for exactly this purpose in corpus.py.
Prints data only -- the coherence judgment is for whoever reads the output.

    python3 experiments/diffusion-searches/diagnostics/diagnose_split_coherence.py \
        --classifierDir checkpoints/diffusion-searches/navigation_classifier/scale/nav_classifier_v4 --maxQueries 2000
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
import os
import pickle
from collections import Counter

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH, loadDescriptions
from train_navigation_classifier import NavigationClassifier, trueChildAt
from evaluate_classifier_branching import scoreChildren, cachePathFor


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/diffusion-searches/navigation_classifier/scale/nav_classifier_v4")
    parser.add_argument("--maxQueries", type=int, default=2000)
    parser.add_argument("--topPairs", type=int, default=6)
    parser.add_argument("--samplesPerChild", type=int, default=6)
    parser.add_argument("--seed", type=int, default=9999)
    return parser.parse_args()


def representativeDescribed(node, corpus, descriptions, limit):
    members = node.memberIndices
    if len(members) == 0:
        return []
    memberVecs = corpus.whitenedMatrix[members]
    centroid = torch.from_numpy(node.centroid.astype(np.float32)).to(corpus.device)
    dist = torch.norm(memberVecs - centroid.unsqueeze(0), dim=1)
    order = dist.argsort().cpu().numpy()
    out = []
    for i in order:
        name = corpus.names[members[i]]
        descs = descriptions.get(name)
        if descs:
            out.append((name, descs[0]))
            if len(out) >= limit:
                break
    return out


def main():
    args = parseArgs()
    device = "cpu"

    with open(os.path.join(args.classifierDir, "config.json")) as f:
        clsConfig = json.load(f)

    whitener = PCAWhitener.load(os.path.join(clsConfig["baseCheckpoint"], "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    hier = HierarchicalClusterIndex.load(clsConfig["treeCache"])
    hier.labelHardness(corpus)
    descriptions = loadDescriptions()

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

    confusion0 = Counter()
    confusion1 = Counter()  # key: (rootChildIdx, trueIdx1, predIdx1)
    depth0Total = depth0Miss = depth1Total = depth1Miss = 0

    for qi, pair in enumerate(queries):
        text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32))
        targetIdx = corpus.nameToIndex[pair["font"]]

        root = hier.root
        trueIdx0, trueChild0 = trueChildAt(root, targetIdx)
        if trueIdx0 is None:
            continue
        probs0 = scoreChildren(model, text, root, device)
        pred0 = int(np.argmax(probs0))
        depth0Total += 1
        if pred0 != trueIdx0:
            depth0Miss += 1
            confusion0[(trueIdx0, pred0)] += 1

        if not trueChild0.children:
            continue
        trueIdx1, trueChild1 = trueChildAt(trueChild0, targetIdx)
        if trueIdx1 is None:
            continue
        probs1 = scoreChildren(model, text, trueChild0, device)
        pred1 = int(np.argmax(probs1))
        depth1Total += 1
        if pred1 != trueIdx1:
            depth1Miss += 1
            confusion1[(trueIdx0, trueIdx1, pred1)] += 1

        if (qi + 1) % 500 == 0 or qi + 1 == len(queries):
            print(f"{qi + 1}/{len(queries)}")

    print(f"\ndepth0 top1 miss rate: {depth0Miss}/{depth0Total} = {depth0Miss / depth0Total:.4f}")
    print(f"depth1 top1 miss rate: {depth1Miss}/{depth1Total} = {depth1Miss / depth1Total:.4f}")

    def printPairs(title, node, pairKeyToChildIdxs, counter, contextLabel):
        print(f"\n=== {title} ===")
        for key, count in counter.most_common(args.topPairs):
            trueIdx, predIdx = pairKeyToChildIdxs(key)
            trueChild = node(key).children[trueIdx]
            predChild = node(key).children[predIdx]
            hardness = node(key).hardness
            print(f"\n-- {contextLabel(key)}: true child {trueIdx} <-> predicted child {predIdx} "
                  f"(confused {count}x), node hardness={hardness:.3f}, "
                  f"sizes: true={len(trueChild.memberIndices)} pred={len(predChild.memberIndices)}")
            print(f"   TRUE child {trueIdx} sample descriptions:")
            for name, desc in representativeDescribed(trueChild, corpus, descriptions, args.samplesPerChild):
                print(f"     [{name}] {desc}")
            print(f"   PREDICTED child {predIdx} sample descriptions:")
            for name, desc in representativeDescribed(predChild, corpus, descriptions, args.samplesPerChild):
                print(f"     [{name}] {desc}")

    printPairs(
        "DEPTH 0 (root) most-confused child pairs",
        lambda key: hier.root,
        lambda key: (key[0], key[1]),
        confusion0,
        lambda key: "root",
    )

    printPairs(
        "DEPTH 1 most-confused child pairs (across all root children)",
        lambda key: hier.root.children[key[0]],
        lambda key: (key[1], key[2]),
        confusion1,
        lambda key: f"root child {key[0]}",
    )


if __name__ == "__main__":
    main()
