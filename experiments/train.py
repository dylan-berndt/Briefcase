"""
Trains the text-conditioned diffusion MLP to predict noise added to
(normalized) font-ViT embeddings, given only the paired text embedding and
timestep. Run after embed_fonts.py and embed_text.py have populated
embeddings/.

    python3 experiments/train.py
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import EmbeddingStats, QueryFontDataset, loadRaw, splitPairs
from diffusion import DiffusionMLP, GaussianDiffusion

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")


def parseArgs():
    parser = argparse.ArgumentParser()
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

    fontEmbeddings, textEmbeddings, pairs = loadRaw()
    print(f"{len(fontEmbeddings)} fonts, {len(textEmbeddings)} unique queries, {len(pairs)} pairs")

    train, test = splitPairs(pairs, testFraction=args.testFraction, seed=args.seed)
    print(f"{len(train)} train pairs, {len(test)} test pairs "
          f"({len({p['font'] for p in train})} / {len({p['font'] for p in test})} fonts)")

    trainFontKeys = sorted({p["font"] for p in train})
    stats = EmbeddingStats.fit(fontEmbeddings, trainFontKeys)

    trainSet = QueryFontDataset(train, fontEmbeddings, textEmbeddings, stats)
    trainLoader = DataLoader(trainSet, batch_size=args.batchSize, shuffle=True,
                              collate_fn=QueryFontDataset.collate, drop_last=True)

    visualDim = next(iter(fontEmbeddings.values())).shape[0]
    textDim = next(iter(textEmbeddings.values())).shape[0]

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
    import json
    with open(os.path.join(CHECKPOINT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "test_pairs.json"), "w") as f:
        json.dump(test, f)

    print(f"Saved checkpoint, normalizer stats, config, and held-out test pairs to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
