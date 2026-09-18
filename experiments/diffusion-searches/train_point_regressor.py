"""
A direct text->embedding point regressor, for ranking WITHIN a leaf
once the classifier has navigated there. The acceptance-set success
criterion is purely geometric (a font's own top-10 nearest neighbors
in the whitened space, independent of text), so the right thing to
rank leaf members by is proximity to a point estimate of "where this
query is pointing," not a query-agnostic centroid or a repurposed
classifier score (measured not to help -- see
rankInLeafByClassifier's negative result). This is the diffusion
model's own conditioning mechanism minus the stochastic sampling
process: a small MLP trained with plain MSE against the true target's
whitened embedding, much cheaper than running reverse diffusion.

    python3 experiments/diffusion-searches/train_point_regressor.py
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from corpus import FontCorpus
from dataset import PCAWhitener, loadRaw, splitQueryCache, EMBEDDINGS_PATH


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


class PointRegressor(nn.Module):
    def __init__(self, textDim, pcaDim, hiddenDim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(textDim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, pcaDim),
        )

    def forward(self, text):
        return self.net(text)


class RegressorDataset(Dataset):
    def __init__(self, pairs, cache, corpus):
        self.examples = []
        for name, vectors in cache.items():
            if name not in corpus.nameToIndex:
                continue
            target = corpus.whitenedMatrix[corpus.nameToIndex[name]].numpy()
            for v in vectors:
                self.examples.append((np.asarray(v, dtype=np.float32), target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        text, target = self.examples[i]
        return {"text": torch.from_numpy(text), "target": torch.from_numpy(target.astype(np.float32))}


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentenceModel", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--baseCheckpoint", default="checkpoints/diffusion_lr_5e-4",
                         help="Only used for whitener.npz / config.json (pcaDim, textDim, testFraction).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batchSize", type=int, default=512)
    parser.add_argument("--learningRate", type=float, default=1e-3)
    parser.add_argument("--hiddenDim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpointDir", default="checkpoints/point_regressor")
    return parser.parse_args()


def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    with open(f"{args.baseCheckpoint}/config.json") as f:
        config = json.load(f)

    fontEmbeddings, sentenceCache = loadRaw(sentenceCachePath=cachePathFor(args.sentenceModel))
    names = sorted(fontEmbeddings.keys())
    trainCache, testCache = splitQueryCache(sentenceCache, names, testFraction=config["testFraction"], seed=config["seed"])
    whitener = PCAWhitener.load(f"{args.baseCheckpoint}/whitener.npz")
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device="cpu")

    trainSet = RegressorDataset(None, trainCache, corpus)
    testSet = RegressorDataset(None, testCache, corpus)
    print(f"{len(trainSet)} train examples, {len(testSet)} test examples")

    trainLoader = DataLoader(trainSet, batch_size=args.batchSize, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=args.batchSize, shuffle=False)

    model = PointRegressor(config["textDim"], config["visualDim"], args.hiddenDim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningRate)

    def evaluate():
        model.eval()
        totalMse, totalCos, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in testLoader:
                text = batch["text"].to(device)
                target = batch["target"].to(device)
                pred = model(text)
                mse = ((pred - target) ** 2).sum(dim=1)
                cos = torch.nn.functional.cosine_similarity(pred, target, dim=1)
                totalMse += mse.sum().item()
                totalCos += cos.sum().item()
                n += text.shape[0]
        return totalMse / n, totalCos / n

    for epoch in range(args.epochs):
        model.train()
        totalLoss, steps = 0.0, 0
        for batch in trainLoader:
            text = batch["text"].to(device)
            target = batch["target"].to(device)
            pred = model(text)
            loss = ((pred - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            steps += 1
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            mse, cos = evaluate()
            print(f"epoch {epoch + 1}/{args.epochs}  trainLoss={totalLoss / steps:.4f}  "
                  f"testMSE={mse:.4f}  testCos={cos:.4f}")

    os.makedirs(args.checkpointDir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.checkpointDir, "checkpoint.pt"))
    with open(os.path.join(args.checkpointDir, "config.json"), "w") as f:
        json.dump({"textDim": config["textDim"], "pcaDim": config["visualDim"], "hiddenDim": args.hiddenDim,
                    "baseCheckpoint": args.baseCheckpoint, "sentenceModel": args.sentenceModel}, f, indent=2)
    print(f"Saved to {args.checkpointDir}")


if __name__ == "__main__":
    main()
