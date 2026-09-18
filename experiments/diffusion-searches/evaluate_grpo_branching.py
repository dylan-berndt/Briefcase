"""
Full-rollout recall@k for a decision-aware GRPO-finetuned checkpoint
(grpo_finetune.py): runs REAL multi-round branching sessions (oracle-
guided, same acceptance-set metric as evaluate_branching.py), but with
the particle batch's conditioning updated at every node transition to
[textEmbedding; currentNode.centroid] -- matching exactly what the model
was trained on. Can't reuse oracle.runSession as-is since it never
changes a session's conditioning after initParticles; this reimplements
the same session loop with that one addition.

    python3 experiments/diffusion-searches/evaluate_grpo_branching.py \
        --checkpointDir checkpoints/diffusion_grpo --maxQueries 60
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from branching import ParticleBatch, advanceTo, findDecisionPoint, previewFonts, pruneAndResample
from oracle import oracleChoice
from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH
from diffusion import DiffusionMLP, GaussianDiffusion

K_VALUES = [1, 5, 10, 50, 100]


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpointDir", required=True)
    parser.add_argument("--maxQueries", type=int, default=60)
    parser.add_argument("--numParticles", type=int, default=40)
    parser.add_argument("--numSteps", type=int, default=50)
    parser.add_argument("--maxModes", type=int, default=4)
    parser.add_argument("--minClusterSize", type=int, default=2)
    parser.add_argument("--persistFor", type=int, default=4)
    parser.add_argument("--maxRounds", type=int, default=5)
    parser.add_argument("--forceAskThreshold", type=int, default=1000)
    parser.add_argument("--checkEvery", type=int, default=1)
    parser.add_argument("--minScheduleFraction", type=float, default=0.1)
    parser.add_argument("--acceptanceSize", type=int, default=10)
    parser.add_argument("--topPerCluster", type=int, default=10)
    parser.add_argument("--numPreviews", type=int, default=5)
    parser.add_argument("--noiseLevels", default="0.0,0.3")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def condFor(textVec, node, device):
    centroid = torch.from_numpy(node.centroid.astype(np.float32)).to(device)
    return torch.cat([textVec, centroid])


def bestRank(finalX, corpus, acceptanceIdx):
    finalX = finalX.detach().to(corpus.device)
    dists = torch.cdist(finalX, corpus.whitenedMatrix)
    bestDists, _ = dists.min(dim=0)
    bestAcceptanceDist = bestDists[acceptanceIdx].min()
    return int((bestDists < bestAcceptanceDist).sum().item())


def runSession(diffusion, model, textVec, acceptanceIdx, corpus, hier, visualDim, numParticles, numSteps,
               maxModes, topPerCluster, numPreviews, checkEvery, minScheduleFraction, minClusterSize,
               persistFor, maxRounds, forceAskThreshold, noiseProb, rng, device):
    steps = diffusion.respacedSteps(numSteps)
    node = hier.root
    cond = condFor(textVec, node, device)
    x = torch.randn(numParticles, visualDim, device=device)
    textBatch = cond.unsqueeze(0).expand(numParticles, -1).contiguous()
    batch = ParticleBatch(x, textBatch, steps, scheduleIndex=len(steps) - 1)

    rounds = 0
    while True:
        if not node.children:
            advanceTo(diffusion, model, batch, 0)
            break

        forceMinChildren = 2 if len(node.memberIndices) > forceAskThreshold else 0
        labels, numQualifying = findDecisionPoint(
            diffusion, model, batch, node, checkEvery=checkEvery, minScheduleFraction=minScheduleFraction,
            maxModes=maxModes, minClusterSize=minClusterSize, persistFor=persistFor,
            forceMinChildren=forceMinChildren)

        if numQualifying == 0:
            break
        atFinalStep = batch.scheduleIndex == 0

        if numQualifying >= 2 and rounds < maxRounds:
            previews, _ = previewFonts(diffusion, model, batch, labels, node, corpus,
                                        topPerCluster=topPerCluster, numPreviews=numPreviews, rng=rng)
            chosen = oracleChoice(previews, acceptanceIdx, corpus, noiseProb, rng)
            rounds += 1
        else:
            nonNegative = labels[labels != -1]
            chosen = int(np.argmax(np.bincount(nonNegative)))

        pruneAndResample(batch, labels, chosen, targetSize=numParticles, rng=rng)
        node = node.children[chosen]
        batch.text = condFor(textVec, node, device).unsqueeze(0).expand(numParticles, -1).contiguous()
        if atFinalStep:
            break

    return batch.x, rounds, node


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.checkpointDir, "config.json")) as f:
        config = json.load(f)
    if not config.get("grpoDecisionAware"):
        print("WARNING: config.json doesn't have grpoDecisionAware=True -- this checkpoint may not "
              "expect [text; nodeCentroid] conditioning. Proceeding anyway.")

    whitener = PCAWhitener.load(os.path.join(args.checkpointDir, "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    hier = HierarchicalClusterIndex.load(config.get("grpoTreeCache", "tree_variant_15_15_5_5_15.pkl"))

    with open(cachePathFor(config["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)
    with open(os.path.join(args.checkpointDir, "test_pairs.json")) as f:
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

    model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"], hiddenDim=config["hiddenDim"],
                          depth=config["depth"], conditioning=config.get("conditioning", "concat")).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpointDir, "checkpoint.pt"), map_location=device))
    model.eval()
    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)

    noiseLevels = [float(x) for x in args.noiseLevels.split(",")]
    for level in noiseLevels:
        hits = {k: 0 for k in K_VALUES}
        roundsList = []
        for qi, pair in enumerate(queries):
            text = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]], dtype=np.float32)).to(device)
            trueIndex = corpus.nameToIndex[pair["font"]]
            _, acceptanceRows = corpus.nearest(corpus.whitenedMatrix[trueIndex].unsqueeze(0), k=args.acceptanceSize)
            acceptanceIdx = [corpus.nameToIndex[n] for n in acceptanceRows[0]]

            querySeed = args.seed * 100003 + qi
            torch.manual_seed(querySeed)
            oracleRng = np.random.default_rng(querySeed)
            finalX, rounds, finalNode = runSession(
                diffusion, model, text, acceptanceIdx, corpus, hier, config["visualDim"],
                args.numParticles, args.numSteps, args.maxModes, args.topPerCluster, args.numPreviews,
                args.checkEvery, args.minScheduleFraction, args.minClusterSize, args.persistFor,
                args.maxRounds, args.forceAskThreshold, level, oracleRng, device)

            rank = bestRank(finalX, corpus, acceptanceIdx)
            for k in K_VALUES:
                if rank < k:
                    hits[k] += 1
            roundsList.append(rounds)
            if (qi + 1) % 100 == 0 or qi + 1 == len(queries):
                print(f"noiseProb={level}: {qi + 1}/{len(queries)}")

        total = len(queries)
        print(f"noiseProb={level}: mean decision points/session={np.mean(roundsList):.2f}")
        for k in K_VALUES:
            print(f"  recall@{k}: {hits[k] / total:.4f}")


if __name__ == "__main__":
    main()
