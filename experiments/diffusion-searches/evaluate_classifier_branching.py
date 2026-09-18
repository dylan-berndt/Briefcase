"""
Full branching-search recall@k driven ENTIRELY by the supervised
navigation classifier (train_navigation_classifier.py) -- no diffusion
sampling at all for decision-making. At each node: score all real
children via the classifier, auto-descend if top-1 confidence clears
--autoThreshold, otherwise present the top min(3, numChildren) candidates
(hard cap per the stated max-3-options constraint) to a simulated oracle
(same acceptance-set-proximity logic as oracle.oracleChoice) and count it
as an asked decision, capped at --maxRounds. At the final leaf, since
there's no generated embedding to rank by, rank the leaf's own real
member fonts by distance to the leaf centroid (a proxy for "most
representative first") and check acceptance-set membership within the
top k, for the standard K_VALUES.

    python3 experiments/diffusion-searches/evaluate_classifier_branching.py \
        --classifierDir checkpoints/nav_classifier --maxQueries 300
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH
from train_navigation_classifier import NavigationClassifier, MAX_CHILDREN

K_VALUES = [1, 5, 10, 50, 100]


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/nav_classifier")
    parser.add_argument("--maxQueries", type=int, default=300)
    parser.add_argument("--maxOptions", type=int, default=3, help="Hard cap on options shown per decision.")
    parser.add_argument("--autoThreshold", type=float, default=1.01,
                         help="Auto-descend without asking when top-1 softmax probability clears this. "
                              "Measured: any threshold below 1.0 (i.e. actually auto-descending on raw "
                              "confidence instead of always using the oracle-consultation budget) makes "
                              "recall@10 WORSE despite using fewer decisions -- default disables auto-descend.")
    parser.add_argument("--maxRounds", type=int, default=5)
    parser.add_argument("--maxDepth", type=int, default=5)
    parser.add_argument("--acceptanceSize", type=int, default=10)
    parser.add_argument("--noiseProb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def scoreChildren(model, text, node, device):
    childCentroids = np.stack([c.centroid for c in node.children]).astype(np.float32)
    numChildren = childCentroids.shape[0]
    padded = np.zeros((MAX_CHILDREN, childCentroids.shape[1]), dtype=np.float32)
    padded[:numChildren] = childCentroids
    mask = np.zeros(MAX_CHILDREN, dtype=bool)
    mask[:numChildren] = True

    textT = text.unsqueeze(0)
    nodeT = torch.from_numpy(node.centroid.astype(np.float32)).unsqueeze(0).to(device)
    childrenT = torch.from_numpy(padded).unsqueeze(0).to(device)
    maskT = torch.from_numpy(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(textT, nodeT, childrenT, maskT).squeeze(0)
    probs = torch.softmax(scores[:numChildren], dim=0)
    return probs.cpu().numpy()


def oracleChoiceAmongChildren(candidateIdx, node, acceptanceIdx, corpus, noiseProb, rng):
    """Picks the candidate child whose members are closest to the acceptance set -- same logic as
    oracle.oracleChoice but scoring real child membership directly (no preview-font sampling needed
    since there's no generation step here)."""
    if rng.random() < noiseProb:
        return rng.choice(candidateIdx)
    acceptanceVecs = corpus.whitenedMatrix[acceptanceIdx]
    best, bestDist = candidateIdx[0], float("inf")
    for i in candidateIdx:
        members = node.children[i].memberIndices
        if len(members) == 0:
            continue
        dist = torch.cdist(corpus.whitenedMatrix[members], acceptanceVecs).min().item()
        if dist < bestDist:
            bestDist, best = dist, i
    return best


def rankInLeaf(node, corpus, acceptanceIdx):
    members = node.memberIndices
    if len(members) == 0:
        return {k: 0 for k in K_VALUES}, 0
    memberVecs = corpus.whitenedMatrix[members]
    centroid = torch.from_numpy(node.centroid.astype(np.float32)).to(corpus.device)
    distToCentroid = torch.norm(memberVecs - centroid.unsqueeze(0), dim=1)
    order = distToCentroid.argsort()
    orderedMembers = [members[i] for i in order.cpu().numpy()]
    acceptanceSet = set(acceptanceIdx)
    hits = {}
    for k in K_VALUES:
        topK = orderedMembers[:k]
        hits[k] = int(any(m in acceptanceSet for m in topK))
    return hits, len(members)


def main():
    args = parseArgs()
    device = "cpu"  # cheap, no need for GPU

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

    hits = {k: 0 for k in K_VALUES}
    roundsList, optionsList, leafSizes = [], [], []

    for qi, pair in enumerate(queries):
        text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32))
        trueIndex = corpus.nameToIndex[pair["font"]]
        _, acceptanceRows = corpus.nearest(corpus.whitenedMatrix[trueIndex].unsqueeze(0), k=args.acceptanceSize)
        acceptanceIdx = [corpus.nameToIndex[n] for n in acceptanceRows[0]]

        oracleRng = np.random.default_rng(args.seed * 100003 + qi)
        node = hier.root
        depth = 0
        rounds = 0
        while node.children and depth < args.maxDepth:
            probs = scoreChildren(model, text, node, device)
            top1 = int(np.argmax(probs))

            if probs[top1] >= args.autoThreshold or rounds >= args.maxRounds:
                chosen = top1
            else:
                k = min(args.maxOptions, len(node.children))
                candidateIdx = np.argsort(-probs)[:k].tolist()
                chosen = oracleChoiceAmongChildren(candidateIdx, node, acceptanceIdx, corpus,
                                                     args.noiseProb, oracleRng)
                optionsList.append(len(candidateIdx))
                rounds += 1

            node = node.children[chosen]
            depth += 1

        roundsList.append(rounds)
        leafHits, leafSize = rankInLeaf(node, corpus, acceptanceIdx)
        leafSizes.append(leafSize)
        for k in K_VALUES:
            hits[k] += leafHits[k]

        print(f"\r{qi + 1}/{len(queries)}", end="")
    print()

    total = len(queries)
    print(f"\nmean decision points/session: {np.mean(roundsList):.2f}  median: {np.median(roundsList):.1f}")
    print(f"mean options/decision: {np.mean(optionsList) if optionsList else 0:.2f}  "
          f"max: {max(optionsList) if optionsList else 0}")
    print(f"final leaf size: mean={np.mean(leafSizes):.1f}  median={np.median(leafSizes):.0f}")
    print(f"\nRecall@k (leaf ranked by distance-to-leaf-centroid), {total} queries:")
    for k in K_VALUES:
        print(f"  recall@{k}: {hits[k] / total:.4f}")


if __name__ == "__main__":
    main()
