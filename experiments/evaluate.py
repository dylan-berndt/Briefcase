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

import numpy as np
import torch

from dataset import EmbeddingStats, loadRaw
from diffusion import DiffusionMLP, GaussianDiffusion

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")
K_VALUES = [1, 5, 10, 50, 100]


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samplesPerQuery", type=int, default=8)
    parser.add_argument("--maxQueries", type=int, default=None,
                         help="Evaluate on a random subset of test queries for a quick check.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    with open(os.path.join(CHECKPOINT_DIR, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(CHECKPOINT_DIR, "test_pairs.json")) as f:
        testPairs = json.load(f)

    stats = EmbeddingStats.load(os.path.join(CHECKPOINT_DIR, "stats.npz"))
    fontEmbeddings, textEmbeddings, _ = loadRaw()

    if args.maxQueries is not None and args.maxQueries < len(testPairs):
        rng = np.random.RandomState(args.seed)
        testPairs = [testPairs[i] for i in rng.choice(len(testPairs), args.maxQueries, replace=False)]

    model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                          hiddenDim=config["hiddenDim"], depth=config["depth"]).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "checkpoint.pt"),
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
        texts = torch.stack([torch.from_numpy(textEmbeddings[p["query"]]) for p in batch]).to(device)
        trueIndices = [nameToIndex[p["font"]] for p in batch]

        # [samplesPerQuery, B, visualDim]
        allSamples = torch.stack([
            diffusion.sample(model, texts, config["visualDim"], device=device)
            for _ in range(args.samplesPerQuery)
        ], dim=0)

        denorm = allSamples.cpu().numpy() * stats.std + stats.mean
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

    print(f"\nRecall@k over {total} held-out test queries, {args.samplesPerQuery} samples/query, "
          f"{len(candidateNames)} candidate fonts:")
    for k in K_VALUES:
        print(f"  recall@{k}: {hits[k] / total:.4f}")


if __name__ == "__main__":
    main()
