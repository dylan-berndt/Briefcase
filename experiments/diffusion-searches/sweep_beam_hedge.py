"""
Joint sweep of beamWidth x hedgeBranchWidth for evaluate_classifier_branching_beam.py,
loading the corpus/model ONCE and reusing them across all combinations (much cheaper
than relaunching the full script per combination). Reports leaf-success (primary and
any-live-branch) for each combo, at a fixed noiseProb/threshold schedule/classifier --
leaf-success only depends on tree-navigation membership, not leaf-ranking, so this
sweep doesn't need the point regressor at all.

    python3 experiments/diffusion-searches/sweep_beam_hedge.py \
        --classifierDir checkpoints/nav_classifier_tree6 --maxDepth 6 \
        --autoThresholds 0.4270,0.8993,0.8692,0.9634,0.9697,0.9966 --noiseProb 0.1
"""
import argparse
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH
from train_navigation_classifier import NavigationClassifier
from evaluate_classifier_branching import cachePathFor
from evaluate_classifier_branching_beam import runSession


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/nav_classifier_tree6")
    parser.add_argument("--maxDepth", type=int, default=6)
    parser.add_argument("--maxOptions", type=int, default=3)
    parser.add_argument("--maxRounds", type=int, default=5)
    parser.add_argument("--acceptanceSize", type=int, default=10)
    parser.add_argument("--autoThresholds", required=True)
    parser.add_argument("--noiseProb", type=float, default=0.1)
    parser.add_argument("--maxQueries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--beamWidths", default="2,3,4,5")
    parser.add_argument("--hedgeBranchWidths", default="1,2")
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
    print(f"{len(queries)} queries, noiseProb={args.noiseProb}")

    model = NavigationClassifier(clsConfig["textDim"], clsConfig["pcaDim"], clsConfig["hiddenDim"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.classifierDir, "checkpoint.pt"), map_location=device))
    model.eval()

    thresholdSchedule = [float(t) for t in args.autoThresholds.split(",")]

    def thresholdAt(depth):
        return thresholdSchedule[min(depth, len(thresholdSchedule) - 1)]

    beamWidths = [int(x) for x in args.beamWidths.split(",")]
    hedgeBranchWidths = [int(x) for x in args.hedgeBranchWidths.split(",")]

    # precompute acceptance sets once per query (independent of beamWidth/hedgeBranchWidth)
    accCache = {}
    for pair in queries:
        trueIndex = corpus.nameToIndex[pair["font"]]
        _, acceptanceRows = corpus.nearest(corpus.whitenedMatrix[trueIndex].unsqueeze(0), k=args.acceptanceSize)
        accCache[pair["font"]] = [corpus.nameToIndex[n] for n in acceptanceRows[0]]

    results = []
    for bw in beamWidths:
        for hbw in hedgeBranchWidths:
            runArgs = SimpleNamespace(maxOptions=args.maxOptions, beamWidth=bw, hedgeBranchWidth=hbw,
                                        maxRounds=args.maxRounds, maxDepth=args.maxDepth,
                                        noiseProb=args.noiseProb)
            primarySuccesses, anySuccesses, poolSizes = [], [], []
            for qi, pair in enumerate(queries):
                text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32))
                trueIndex = corpus.nameToIndex[pair["font"]]
                acceptanceIdx = accCache[pair["font"]]
                oracleRng = np.random.default_rng(args.seed * 100003 + qi)
                beam, rounds, optionsShown = runSession(model, text, hier, corpus, acceptanceIdx, runArgs,
                                                           oracleRng, device, thresholdAt)
                primaryNode = beam[0]["node"]
                primarySuccesses.append(int(trueIndex in set(primaryNode.memberIndices.tolist())))
                anySuccesses.append(int(any(trueIndex in set(e["node"].memberIndices.tolist()) for e in beam)))
                pooled = set()
                for e in beam:
                    pooled.update(e["node"].memberIndices.tolist())
                poolSizes.append(len(pooled))
            primaryRate = float(np.mean(primarySuccesses))
            anyRate = float(np.mean(anySuccesses))
            meanPool = float(np.mean(poolSizes))
            medianPool = float(np.median(poolSizes))
            results.append((bw, hbw, primaryRate, anyRate, meanPool, medianPool))
            print(f"beamWidth={bw}  hedgeBranchWidth={hbw}  primary-leaf-success={primaryRate:.4f}  "
                  f"any-leaf-success={anyRate:.4f}  pooled-set: mean={meanPool:.1f} median={medianPool:.0f}")

    best = max(results, key=lambda r: r[3])
    print(f"\nBest by any-leaf-success alone: beamWidth={best[0]}  hedgeBranchWidth={best[1]}  "
          f"any-leaf-success={best[3]:.4f}  pooled-set mean={best[4]:.1f}")
    print("(Choose the actual config by weighing leaf-success against pooled-set size -- "
          "a bigger pool isn't free, it's a bigger final result set shown to the user.)")


if __name__ == "__main__":
    main()
