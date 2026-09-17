"""
Separates two different questions the earlier diagnostics conflated:
does a generated sample look like the RIGHT font (conditioning
correctness), vs does it look like ANY plausible font AT ALL (marginal
generation quality/realism)? diagnose_text_ablation.py showed text
conditioning has a real but modest effect on the per-step denoising
loss; diagnose_conditioning.py showed generated samples average only
~0.13 cosine to the corpus-wide mean font vs real fonts' ~0.75, and have
much higher pairwise-similarity variance than real fonts do -- both
consistent with generation drifting off the real manifold entirely, not
just landing on the wrong point on it.

For each of a batch of held-out queries, generates one sample and finds
its nearest neighbor by cosine similarity across the ENTIRE font corpus
(regardless of whether that neighbor is the query's true target), then
compares that "how close is this to the closest real font" distribution
against the same nearest-neighbor-in-corpus distribution computed for
real font embeddings themselves (leave-one-out) as a realism baseline.

    python3 experiments/diagnose_manifold_realism.py --numQueries 500
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from dataset import PCAWhitener, loadFontEmbeddings, EMBEDDINGS_PATH
from diffusion import DiffusionMLP, GaussianDiffusion

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numQueries", type=int, default=500)
    parser.add_argument("--evalSteps", type=int, default=50)
    parser.add_argument("--realismSample", type=int, default=2000,
                         help="Real fonts sampled for the leave-one-out nearest-neighbor baseline.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    with open(os.path.join(CHECKPOINT_DIR, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(CHECKPOINT_DIR, "test_pairs.json")) as f:
        testPairs = json.load(f)

    whitener = PCAWhitener.load(os.path.join(CHECKPOINT_DIR, "whitener.npz"))
    fontEmbeddings = loadFontEmbeddings(EMBEDDINGS_PATH)
    with open(cachePathFor(config["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)

    testPairs = [p for p in testPairs if p["font"] in fontEmbeddings]
    byFont = {}
    for p in testPairs:
        byFont.setdefault(p["font"], p)
    batch = list(byFont.values())
    rng.shuffle(batch)
    batch = batch[:args.numQueries]

    candidateNames = sorted(fontEmbeddings.keys())
    candidateMatrix = np.stack([fontEmbeddings[n] for n in candidateNames])
    candidateMatrix = candidateMatrix / (np.linalg.norm(candidateMatrix, axis=1, keepdims=True) + 1e-8)
    candidateMatrixT = torch.from_numpy(candidateMatrix.astype(np.float32)).to(device)
    nameToIndex = {n: i for i, n in enumerate(candidateNames)}

    model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                          hiddenDim=config["hiddenDim"], depth=config["depth"],
                          conditioning=config.get("conditioning", "concat")).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "checkpoint.pt"), map_location=device))
    model.eval()
    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)

    texts = torch.stack([
        torch.from_numpy(np.asarray(sentenceCache[p["font"]][p["index"]], dtype=np.float32))
        for p in batch
    ]).to(device)
    trueIndices = torch.tensor([nameToIndex[p["font"]] for p in batch], device=device)

    generated = diffusion.sample(model, texts, config["visualDim"], device=device, numSteps=args.evalSteps)
    generated = whitener.inverseTransform(generated.cpu().numpy())
    generated = generated / (np.linalg.norm(generated, axis=-1, keepdims=True) + 1e-8)
    generatedT = torch.from_numpy(generated.astype(np.float32)).to(device)

    genSims = generatedT @ candidateMatrixT.T  # [numQueries, numCandidates]
    genNearest, genNearestIdx = genSims.max(dim=1)
    genTrueTargetSim = genSims.gather(1, trueIndices.unsqueeze(1)).squeeze(1)
    genNearestIsTrueTarget = (genNearestIdx == trueIndices).float().mean().item()

    realismIdx = rng.choice(len(candidateNames), min(args.realismSample, len(candidateNames)), replace=False)
    realismMatrix = candidateMatrixT[realismIdx]
    realSims = realismMatrix @ candidateMatrixT.T
    realSims[torch.arange(len(realismIdx)), torch.from_numpy(realismIdx).to(device)] = -1.0  # exclude self
    realNearest, _ = realSims.max(dim=1)

    print(f"{len(batch)} generated samples ({args.evalSteps} steps), "
          f"{len(realismIdx)}-real-font leave-one-out baseline\n")
    print(f"generated -> nearest-in-corpus cosine:      mean {genNearest.mean().item():.4f}  "
          f"std {genNearest.std().item():.4f}  min {genNearest.min().item():.4f}  max {genNearest.max().item():.4f}")
    print(f"real font -> nearest-other-real cosine:     mean {realNearest.mean().item():.4f}  "
          f"std {realNearest.std().item():.4f}  min {realNearest.min().item():.4f}  max {realNearest.max().item():.4f}")
    print(f"generated -> own true-target cosine:        mean {genTrueTargetSim.mean().item():.4f}  "
          f"std {genTrueTargetSim.std().item():.4f}")
    print(f"fraction where nearest-in-corpus IS the true target: {genNearestIsTrueTarget:.4f}")


if __name__ == "__main__":
    main()
