"""
Tests a specific hypothesis before writing any new search mechanism:
when the classifier picks the WRONG branch at some tree level (the
true child isn't in its top-3), is the true target usually still
GEOMETRICALLY close to the wrongly-favored branch (just across an
arbitrary tree boundary), or is it actually far away in embedding
space?

This matters because it decides whether "free the search from rigid
tree topology" (consider candidates by embedding-space proximity to a
running belief, not strict tree descendance) is even geometrically
possible to pay off. If wrong turns are near-misses, a proximity-aware
search should recover often. If wrong turns are far misses, there's
nothing nearby to recover into, and de-tree-ifying the search wouldn't
help.

For each held-out query, walks the TRUE path (ground truth via
trueChildAt) using the classifier's own top-1 pick as "what the
search would actually do" at each level. At the FIRST level where the
classifier's own top-3 misses the true child, records:
  - trueToChosen: distance from the target's own embedding to the
    WRONGLY-favored (top-1) child's centroid.
  - trueToCorrect: distance from the target's own embedding to the
    ACTUAL correct child's own centroid (the floor -- how close the
    target naturally sits to ITS OWN region).
  - trueToRandomOther: distance to a random OTHER sibling's centroid,
    as a baseline for "how close is close."
Reports the distribution of these three, and specifically what
fraction of wrong turns have trueToChosen close to trueToCorrect
(i.e., the wrongly-favored branch is nearly as close as the correct
one) vs. close to trueToRandomOther (i.e., no better than a random
wrong guess).

    python3 experiments/diffusion-searches/diagnostics/diagnose_wrong_turn_geometry.py \
        --classifierDir checkpoints/diffusion-searches/navigation_classifier/scale/nav_classifier_v4 --maxQueries 1000
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

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH
from train_navigation_classifier import NavigationClassifier, trueChildAt
from evaluate_classifier_branching import scoreChildren, cachePathFor


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/diffusion-searches/navigation_classifier/scale/nav_classifier_v4")
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

    trueToChosen, trueToCorrect, trueToRandom = [], [], []
    missDepths = []
    trueRanks = []

    for qi, pair in enumerate(queries):
        text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32))
        targetIdx = corpus.nameToIndex[pair["font"]]
        targetVec = corpus.whitenedMatrix[targetIdx]

        node = hier.root
        depth = 0
        while node.children and depth < args.maxDepth:
            trueIdx, trueChild = trueChildAt(node, targetIdx)
            if trueIdx is None:
                break
            probs = scoreChildren(model, text, node, device)
            k = min(args.maxOptions, len(node.children))
            top3 = np.argsort(-probs)[:k].tolist()

            if trueIdx not in top3:
                chosenIdx = int(np.argmax(probs))  # what the search would actually favor
                chosenCentroid = torch.from_numpy(node.children[chosenIdx].centroid.astype(np.float32))
                correctCentroid = torch.from_numpy(node.children[trueIdx].centroid.astype(np.float32))
                fullRanking = np.argsort(-probs).tolist()
                trueRanks.append((depth, fullRanking.index(trueIdx) + 1))  # 1-indexed rank

                others = [i for i in range(len(node.children)) if i not in (chosenIdx, trueIdx)]
                if others:
                    randomIdx = rng.choice(others)
                    randomCentroid = torch.from_numpy(node.children[randomIdx].centroid.astype(np.float32))
                    trueToRandom.append(torch.norm(targetVec - randomCentroid).item())

                trueToChosen.append(torch.norm(targetVec - chosenCentroid).item())
                trueToCorrect.append(torch.norm(targetVec - correctCentroid).item())
                missDepths.append(depth)
                break  # only the FIRST miss per query, matching diagnose_top3_miss.py's convention

            node = trueChild
            depth += 1

        if (qi + 1) % 200 == 0 or qi + 1 == len(queries):
            print(f"{qi + 1}/{len(queries)}")

    trueToChosen = np.array(trueToChosen)
    trueToCorrect = np.array(trueToCorrect)
    trueToRandom = np.array(trueToRandom)
    missDepths = np.array(missDepths)

    print(f"\n{len(trueToChosen)} first-misses out of {len(queries)} queries")
    print(f"distance from target's own embedding to:")
    print(f"  the CORRECT branch's centroid (floor):       mean={trueToCorrect.mean():.3f}  median={np.median(trueToCorrect):.3f}")
    print(f"  the WRONGLY-favored branch's centroid:       mean={trueToChosen.mean():.3f}  median={np.median(trueToChosen):.3f}")
    print(f"  a RANDOM other sibling's centroid (ceiling): mean={trueToRandom.mean():.3f}  median={np.median(trueToRandom):.3f}")

    # normalize: 0 = as close as the correct branch, 1 = as far as a random sibling
    denom = trueToRandom - trueToCorrect
    denom[denom == 0] = 1e-9
    normalized = (trueToChosen - trueToCorrect) / denom
    normalized = np.clip(normalized, 0, 2)
    print(f"\nnormalized position of the wrongly-favored branch between 'correct' (0) and 'random' (1):")
    print(f"  mean={normalized.mean():.3f}  median={np.median(normalized):.3f}")
    print(f"  fraction of wrong turns where the wrong branch is CLOSER to correct than to random (<0.5): "
          f"{(normalized < 0.5).mean():.4f}")
    print(f"  fraction essentially AS FAR as random (>=0.9): {(normalized >= 0.9).mean():.4f}")

    print("\nby first-miss depth:")
    for d in sorted(set(missDepths.tolist())):
        m = missDepths == d
        print(f"  depth {d}: n={m.sum()}  mean normalized position={normalized[m].mean():.3f}")

    print("\ntrue child's actual rank by classifier score, when it misses top-3, by depth:")
    for d in sorted({r[0] for r in trueRanks}):
        ranks = np.array([r[1] for r in trueRanks if r[0] == d])
        within5 = (ranks <= 5).mean()
        within8 = (ranks <= 8).mean()
        print(f"  depth {d}: n={len(ranks)}  mean rank={ranks.mean():.1f}  median rank={np.median(ranks):.0f}  "
              f"within top-5={within5:.3f}  within top-8={within8:.3f}")


if __name__ == "__main__":
    main()
