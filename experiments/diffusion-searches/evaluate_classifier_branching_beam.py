"""
Beam-search variant of evaluate_classifier_branching.py. The oracle
interaction is unchanged (one active/"primary" branch is shown <=3
classifier-ranked options per round, the oracle -- possibly noisy --
clicks exactly one, same as before). What changes: the non-chosen
candidates from that same round are NOT discarded. They're kept alive
as "hedge" branches, weighted by the classifier's own probability for
them, and auto-advanced each subsequent round via the classifier's own
top-1 choice (no further oracle interaction, since a real user only
clicks once per round). At the end, recall@k is checked against the
UNION of all live branches' leaf members, ordered primary-branch-first
then by descending hedge weight -- so a single wrong oracle click no
longer permanently forecloses the true target, as long as the
classifier ranked the correct child among the shown options.

This targets the noise-sensitivity gap measured in
evaluate_classifier_branching.py (recall@10 drops from 0.66 at
noiseProb=0 to 0.48/0.37 at noiseProb=0.1/0.2) -- a real, not
hypothetical, concern per the user: the oracle here approximates a
real user, and 0.1-0.2 is a *lower* bound on realistic error rates.

    python3 experiments/diffusion-searches/evaluate_classifier_branching_beam.py \
        --classifierDir checkpoints/nav_classifier_v4 --beamWidth 3 --noiseProb 0.1
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH
from train_navigation_classifier import NavigationClassifier
from evaluate_classifier_branching import scoreChildren, oracleChoiceAmongChildren, cachePathFor, K_VALUES


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/nav_classifier_v4")
    parser.add_argument("--maxQueries", type=int, default=300)
    parser.add_argument("--maxOptions", type=int, default=3)
    parser.add_argument("--beamWidth", type=int, default=3,
                         help="Max live branches (1 primary + beamWidth-1 hedges). beamWidth=1 "
                              "reduces exactly to evaluate_classifier_branching.py's hard-commit behavior.")
    parser.add_argument("--hedgeBranchWidth", type=int, default=1,
                         help="How many of a hedge branch's own top children to spawn as new candidates "
                              "each round (default 1 = collapse to top-1, original behavior). >1 lets "
                              "hedge branches genuinely widen the search instead of each tracking a single "
                              "path -- global pruning to beamWidth still applies afterward.")
    parser.add_argument("--maxRounds", type=int, default=5)
    parser.add_argument("--maxDepth", type=int, default=5)
    parser.add_argument("--acceptanceSize", type=int, default=10)
    parser.add_argument("--noiseProb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def runSession(model, text, hier, corpus, acceptanceIdx, args, oracleRng, device):
    beam = [{"node": hier.root, "weight": 1.0, "isPrimary": True}]
    depth = 0
    rounds = 0
    optionsShown = []

    while depth < args.maxDepth:
        newBeam = []
        primary = beam[0]
        node = primary["node"]
        if node.children:
            probs = scoreChildren(model, text, node, device)
            k = min(args.maxOptions, len(node.children))
            candidateIdx = np.argsort(-probs)[:k].tolist()
            chosen = oracleChoiceAmongChildren(candidateIdx, node, acceptanceIdx, corpus,
                                                 args.noiseProb, oracleRng)
            optionsShown.append(len(candidateIdx))
            rounds += 1
            newBeam.append({"node": node.children[chosen], "weight": 1.0, "isPrimary": True})
            for c in candidateIdx:
                if c != chosen:
                    newBeam.append({"node": node.children[c], "weight": float(probs[c]), "isPrimary": False})
        else:
            newBeam.append(primary)

        for entry in beam[1:]:
            node = entry["node"]
            if node.children:
                probs = scoreChildren(model, text, node, device)
                kh = min(args.hedgeBranchWidth, len(node.children))
                topIdx = np.argsort(-probs)[:kh].tolist()
                for c in topIdx:
                    newBeam.append({"node": node.children[c],
                                     "weight": entry["weight"] * float(probs[c]),
                                     "isPrimary": False})
            else:
                newBeam.append(entry)

        primaryEntries = [e for e in newBeam if e["isPrimary"]]
        others = sorted([e for e in newBeam if not e["isPrimary"]], key=lambda e: -e["weight"])
        beam = primaryEntries[:1] + others[:max(0, args.beamWidth - 1)]
        depth += 1

    return beam, rounds, optionsShown


def rankPooledBeam(beam, corpus, acceptanceIdx):
    acceptanceSet = set(acceptanceIdx)
    ordered = sorted(beam, key=lambda e: (0 if e["isPrimary"] else 1, -e["weight"]))
    orderedMembers = []
    seen = set()
    for entry in ordered:
        members = entry["node"].memberIndices
        if len(members) == 0:
            continue
        memberVecs = corpus.whitenedMatrix[members]
        centroid = torch.from_numpy(entry["node"].centroid.astype(np.float32)).to(corpus.device)
        dist = torch.norm(memberVecs - centroid.unsqueeze(0), dim=1)
        order = dist.argsort()
        for i in order.cpu().numpy():
            m = members[i]
            if m not in seen:
                seen.add(m)
                orderedMembers.append(m)
    hits = {}
    for k in K_VALUES:
        topK = orderedMembers[:k]
        hits[k] = int(any(m in acceptanceSet for m in topK))
    return hits, len(orderedMembers)


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
    print(f"{len(queries)} queries, beamWidth={args.beamWidth}, noiseProb={args.noiseProb}")

    model = NavigationClassifier(clsConfig["textDim"], clsConfig["pcaDim"], clsConfig["hiddenDim"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.classifierDir, "checkpoint.pt"), map_location=device))
    model.eval()

    hits = {k: 0 for k in K_VALUES}
    roundsList, optionsList, poolSizes = [], [], []

    for qi, pair in enumerate(queries):
        text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32))
        trueIndex = corpus.nameToIndex[pair["font"]]
        _, acceptanceRows = corpus.nearest(corpus.whitenedMatrix[trueIndex].unsqueeze(0), k=args.acceptanceSize)
        acceptanceIdx = [corpus.nameToIndex[n] for n in acceptanceRows[0]]

        oracleRng = np.random.default_rng(args.seed * 100003 + qi)
        beam, rounds, optionsShown = runSession(model, text, hier, corpus, acceptanceIdx, args, oracleRng, device)

        roundsList.append(rounds)
        optionsList.extend(optionsShown)
        leafHits, poolSize = rankPooledBeam(beam, corpus, acceptanceIdx)
        poolSizes.append(poolSize)
        for k in K_VALUES:
            hits[k] += leafHits[k]

        if (qi + 1) % 100 == 0 or qi + 1 == len(queries):
            print(f"{qi + 1}/{len(queries)}")

    total = len(queries)
    print(f"\nmean decision points/session: {np.mean(roundsList):.2f}  median: {np.median(roundsList):.1f}")
    print(f"mean options/decision: {np.mean(optionsList) if optionsList else 0:.2f}  "
          f"max: {max(optionsList) if optionsList else 0}")
    print(f"pooled candidate set size: mean={np.mean(poolSizes):.1f}  median={np.median(poolSizes):.0f}")
    print(f"\nRecall@k (beam-pooled, primary-branch-first), {total} queries:")
    for k in K_VALUES:
        print(f"  recall@{k}: {hits[k] / total:.4f}")


if __name__ == "__main__":
    main()
