"""
Trains the text-conditioned diffusion MLP to predict noise added to
(normalized) font embeddings, given only the paired query embedding and
timestep. Run after embed_text_local.py has produced a sentence cache.

    python3 experiments/train.py --sentenceModel BAAI/bge-large-en-v1.5
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import EmbeddingStats, QueryFontDataset, loadRaw, splitQueryCache
from diffusion import DiffusionMLP, GaussianDiffusion

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentenceModel", default="BAAI/bge-large-en-v1.5",
                         help="Must match what embed_text_local.py was run with.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batchSize", type=int, default=256)
    parser.add_argument("--learningRate", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hiddenDim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--testFraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    fontEmbeddings, sentenceCache = loadRaw(sentenceCachePath=cachePathFor(args.sentenceModel))
    print(f"{len(fontEmbeddings)} matched fonts")

    names = sorted(fontEmbeddings.keys())
    trainCache, testCache = splitQueryCache(sentenceCache, names, testFraction=args.testFraction, seed=args.seed)
    print(f"{sum(len(v) for v in trainCache.values())} train queries, "
          f"{sum(len(v) for v in testCache.values())} test queries, {len(names)} fonts (never held out)")

    stats = EmbeddingStats.fit(fontEmbeddings, names)

    trainSet = QueryFontDataset(trainCache, fontEmbeddings, stats)
    trainLoader = DataLoader(trainSet, batch_size=args.batchSize, shuffle=True,
                              collate_fn=QueryFontDataset.collate, drop_last=True)

    visualDim = next(iter(fontEmbeddings.values())).shape[0]
    textDim = next(iter(sentenceCache.values())).shape[-1]

    model = DiffusionMLP(visualDim=visualDim, textDim=textDim, hiddenDim=args.hiddenDim,
                          depth=args.depth).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningRate)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        totalLoss, steps = 0.0, 0
        for batch in trainLoader:
            text = batch["text"].to(device)
            visual = batch["visual"].to(device)

            loss = diffusion.trainingLoss(model, visual, text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
            steps += 1

        print(f"epoch {epoch + 1}/{args.epochs}  loss {totalLoss / max(steps, 1):.5f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "checkpoint.pt"))
    stats.save(os.path.join(CHECKPOINT_DIR, "stats.npz"))

    config = vars(args) | {"visualDim": visualDim, "textDim": textDim}
    with open(os.path.join(CHECKPOINT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # test split as (font, vector) pairs -- evaluate.py re-reads the same
    # sentence cache and pulls each vector out by index at eval time.
    testPairs = [{"font": name, "index": i} for name, vectors in testCache.items() for i in range(len(vectors))]
    with open(os.path.join(CHECKPOINT_DIR, "test_pairs.json"), "w") as f:
        json.dump(testPairs, f)

    print(f"Saved checkpoint, normalizer stats, config, and held-out test pairs to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
