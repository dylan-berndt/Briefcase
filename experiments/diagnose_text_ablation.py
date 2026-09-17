"""
Directly tests whether the trained diffusion model's noise prediction
actually depends on the text conditioning, without going through iterative
sampling at all: fixes x0, the sampled noise, and t, then computes the
denoising loss twice -- once with each example's real paired text
embedding, once with those same embeddings shuffled across the batch (so
every example gets a mismatched, unrelated query) -- and reports both,
stratified by timestep bucket (low-t/low-noise steps are close to a
trivial copy task regardless of conditioning; high-t/high-noise steps are
where conditioning should matter most for determining which font gets
generated). If real and shuffled loss are close at a given noise level,
the model isn't using text there.

    python3 experiments/diagnose_text_ablation.py --numSamples 4000
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
T_BUCKET_EDGES = [0, 100, 300, 500, 700, 900, 1000]


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numSamples", type=int, default=4000)
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
    if args.numSamples < len(testPairs):
        idx = rng.choice(len(testPairs), args.numSamples, replace=False)
        testPairs = [testPairs[i] for i in idx]

    model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                          hiddenDim=config["hiddenDim"], depth=config["depth"]).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "checkpoint.pt"), map_location=device))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)

    x0 = torch.stack([
        torch.from_numpy(whitener.transform(fontEmbeddings[p["font"]]).astype(np.float32))
        for p in testPairs
    ]).to(device)
    text = torch.stack([
        torch.from_numpy(np.asarray(sentenceCache[p["font"]][p["index"]], dtype=np.float32))
        for p in testPairs
    ]).to(device)

    shuffledText = text[torch.randperm(len(testPairs), device=device)]

    print(f"{len(testPairs)} samples\n")
    print(f"{'t range':>12s}  {'n':>6s}  {'loss (real text)':>17s}  {'loss (shuffled text)':>21s}")

    with torch.no_grad():
        for lo, hi in zip(T_BUCKET_EDGES[:-1], T_BUCKET_EDGES[1:]):
            tScale = config["timesteps"] / 1000
            loT, hiT = int(lo * tScale), int(hi * tScale)
            t = torch.randint(loT, hiT, (len(testPairs),), device=device)
            xt, noise = diffusion.qSample(x0, t)

            realLoss = torch.nn.functional.mse_loss(model(xt, t, text), noise).item()
            shuffledLoss = torch.nn.functional.mse_loss(model(xt, t, shuffledText), noise).item()

            print(f"  [{loT:>4d},{hiT:>4d})  {len(testPairs):>6d}  {realLoss:>17.5f}  {shuffledLoss:>21.5f}")


if __name__ == "__main__":
    main()
