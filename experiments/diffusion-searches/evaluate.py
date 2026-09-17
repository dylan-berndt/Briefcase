"""
Recall@k over the held-out text prompts, run through the full reverse
diffusion process (no shortcuts / no fixed-t denoising). For each held-out
query, draws N independent reverse-diffusion samples (best-of-N), ranks
ALL known fonts (fonts are never held out) by the best cosine similarity
across those N samples, and checks whether the query's true font lands in
the top k.

    python3 experiments/evaluate.py --samplesPerQuery 8
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from diffusion import DiffusionMLP, TwoPhaseDiffusionMLP, GaussianDiffusion
from dataset import (PCAWhitener, TextCenterer, loadFontEmbeddings, EMBEDDINGS_PATH,
                      loadTagPresenceCache, concatenateTagPresence,
                      loadTfidfFeatureCache, concatenateTfidfCache)

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")
K_VALUES = [1, 5, 10, 50, 100]


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samplesPerQuery", type=int, default=8)
    parser.add_argument("--evalSteps", type=int, default=50,
                         help="Reverse-diffusion steps to sample with at eval time -- can be less than "
                              "the training schedule's --timesteps (default 1000) to match a faster "
                              "production sampling budget. Uses the same respacing trick regardless of "
                              "how many steps training used, so this never requires retraining.")
    parser.add_argument("--maxQueries", type=int, default=None,
                         help="Evaluate on a random subset of test queries for a quick check.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpointDir", default=CHECKPOINT_DIR,
                         help="Point at a variant trained with train.py --checkpointDir to compare "
                              "against the default (e.g. a --conditioning film run).")
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    with open(os.path.join(args.checkpointDir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.checkpointDir, "test_pairs.json")) as f:
        testPairs = json.load(f)

    whitener = PCAWhitener.load(os.path.join(args.checkpointDir, "whitener.npz"))
    fontEmbeddings = loadFontEmbeddings(EMBEDDINGS_PATH)
    with open(cachePathFor(config["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)
    if config.get("centerText"):
        centerer = TextCenterer.load(os.path.join(args.checkpointDir, "textCenter.npz"))
        sentenceCache = centerer.applyToCache(sentenceCache)
    if config.get("tagConditioning"):
        tagCache = loadTagPresenceCache()
        sentenceCache = concatenateTagPresence(sentenceCache, tagCache)
    if config.get("tfidfDim", 0) > 0:
        tfidfCache = loadTfidfFeatureCache(config["tfidfDim"])
        sentenceCache = concatenateTfidfCache(sentenceCache, tfidfCache)

    # Only pairs whose font still has a visual embedding (matches dataset.py's
    # matched set at train time) are valid candidates for eval too.
    testPairs = [p for p in testPairs if p["font"] in fontEmbeddings]

    if args.maxQueries is not None and args.maxQueries < len(testPairs):
        rng = np.random.RandomState(args.seed)
        testPairs = [testPairs[i] for i in rng.choice(len(testPairs), args.maxQueries, replace=False)]

    if config.get("architecture", "standard") == "shapeI":
        model = TwoPhaseDiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                                      hiddenDim=config["hiddenDim"], timeDim=config.get("timeDim", 64),
                                      numPlainBlocks=config.get("numPlainBlocks", 1),
                                      numConditionedBlocks=config.get("numConditionedBlocks", 2)).to(device)
    else:
        model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                              hiddenDim=config["hiddenDim"], depth=config["depth"],
                              conditioning=config.get("conditioning", "concat")).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpointDir, "checkpoint.pt"),
                                      map_location=device))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)

    # Candidate set = every font we have a visual embedding for. Fonts are
    # never held out, so the query's true font is guaranteed to be in here.
    candidateNames = sorted(fontEmbeddings.keys())
    candidateMatrix = np.stack([fontEmbeddings[name] for name in candidateNames], axis=0)
    candidateMatrix = candidateMatrix / (np.linalg.norm(candidateMatrix, axis=1, keepdims=True) + 1e-8)
    candidateMatrix = torch.from_numpy(candidateMatrix.astype(np.float32)).to(device)
    nameToIndex = {name: i for i, name in enumerate(candidateNames)}

    hits = {k: 0 for k in K_VALUES}
    total = 0

    batchSize = 32
    for start in range(0, len(testPairs), batchSize):
        batch = testPairs[start: start + batchSize]
        texts = torch.stack([
            torch.from_numpy(np.asarray(sentenceCache[p["font"]][p["index"]], dtype=np.float32))
            for p in batch
        ]).to(device)
        trueIndices = [nameToIndex[p["font"]] for p in batch]

        # [samplesPerQuery, B, visualDim]
        allSamples = torch.stack([
            diffusion.sample(model, texts, config["visualDim"], device=device, numSteps=args.evalSteps)
            for _ in range(args.samplesPerQuery)
        ], dim=0)

        denorm = whitener.inverseTransform(allSamples.cpu().numpy())
        denorm = denorm / (np.linalg.norm(denorm, axis=-1, keepdims=True) + 1e-8)
        denorm = torch.from_numpy(denorm.astype(np.float32)).to(device)

        # best-of-N: max cosine similarity across the N samples per query
        # denorm: [N, B, D], candidateMatrix: [C, D] -> sims: [N, B, C]
        sims = torch.einsum("nbd,cd->nbc", denorm, candidateMatrix)
        bestSims, _ = sims.max(dim=0)  # [B, C]

        ranking = bestSims.argsort(dim=1, descending=True)  # [B, C]
        for i, trueIndex in enumerate(trueIndices):
            rank = (ranking[i] == trueIndex).nonzero(as_tuple=True)[0].item()
            for k in K_VALUES:
                if rank < k:
                    hits[k] += 1
        total += len(batch)

        print(f"\r{min(start + batchSize, len(testPairs))}/{len(testPairs)} queries evaluated", end="")
    print()

    actualSteps = min(args.evalSteps, config["timesteps"])
    print(f"\nRecall@k over {total} held-out test queries, {args.samplesPerQuery} samples/query, "
          f"{actualSteps}/{config['timesteps']} reverse-diffusion steps, "
          f"{len(candidateNames)} candidate fonts:")
    for k in K_VALUES:
        print(f"  recall@{k}: {hits[k] / total:.4f}")


if __name__ == "__main__":
    main()
