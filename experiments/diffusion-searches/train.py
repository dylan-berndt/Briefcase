"""
Trains the text-conditioned diffusion MLP to predict noise added to a
PCA-whitened, reduced-dimensional projection of the font embeddings
(see dataset.PCAWhitener), given only the paired query embedding and
timestep. Run after embed_text_local.py has produced a sentence cache.

    python3 experiments/train.py --sentenceModel BAAI/bge-large-en-v1.5
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import (PCAWhitener, QueryFontDataset, TextCenterer, loadRaw, splitQueryCache,
                      loadTagPresenceCache, concatenateTagPresence,
                      loadTfidfFeatureCache, concatenateTfidfCache)
from diffusion import DiffusionMLP, TwoPhaseDiffusionMLP, GaussianDiffusion

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentenceModel", default="BAAI/bge-large-en-v1.5",
                         help="Must match what embed_text_local.py was run with.")
    parser.add_argument("--pcaDim", type=int, default=64,
                         help="Dimensionality of the whitened PCA subspace the diffusion process "
                              "actually operates in (embeddings/all.json's own effective rank is "
                              "~57.6/512 -- 64 keeps a little headroom above that for differentiation).")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batchSize", type=int, default=256)
    parser.add_argument("--numWorkers", type=int, default=0,
                         help="DataLoader worker processes. 0 (default, and what every run so far actually "
                              "used) loads synchronously in the main process -- a likely real bottleneck for "
                              "a model this small, where each GPU step is near-instant and the Python-side "
                              "batch fetch/collate dominates wall-clock time.")
    parser.add_argument("--learningRate", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hiddenDim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--conditioning", choices=["concat", "film"], default="concat",
                         help="See diffusion.ResidualBlock's docstring. 'film' (Perez et al. 2018) lets "
                              "the text/timestep condition modulate the hidden state's own scale/shift "
                              "instead of just being added alongside it -- untested prior to this run.")
    parser.add_argument("--architecture", choices=["standard", "shapeI"], default="standard",
                         help="'standard' = DiffusionMLP (--hiddenDim/--depth/--conditioning apply). "
                              "'shapeI' = TwoPhaseDiffusionMLP: a much smaller network (see "
                              "--numPlainBlocks/--numConditionedBlocks/--hiddenDim) where plain residual "
                              "blocks process the noised input alone first, then a second group of blocks "
                              "injects timestep+text conditioning via simple additive projection (no FiLM) "
                              "-- --conditioning/--depth are ignored for this architecture.")
    parser.add_argument("--numPlainBlocks", type=int, default=1,
                         help="shapeI only: unconditioned blocks in phase 1.")
    parser.add_argument("--numConditionedBlocks", type=int, default=2,
                         help="shapeI only: conditioned blocks in phase 2.")
    parser.add_argument("--timeDim", type=int, default=64, help="shapeI only (standard hardcodes 128).")
    parser.add_argument("--centerText", action="store_true",
                         help="Mean-center + renormalize text query embeddings before use (see "
                              "dataset.TextCenterer) -- fixes measured anisotropy in the raw BGE space "
                              "that compresses unrelated fonts' cosine similarity near 0.86. Fit on "
                              "training queries only, saved to textCenter.npz for eval-time reuse.")
    parser.add_argument("--tagConditioning", action="store_true",
                         help="Concatenate a sparse multi-hot tag-presence vector (tag_conditioning.py, "
                              "matched against the existing retrieval-head vocab) onto the dense text "
                              "embedding -- makes tag PRESENCE the explicit representational unit instead "
                              "of relying on a dense embedding to implicitly encode it. Requires "
                              "embeddings/tagPresence.pkl (build_tag_presence_cache.py).")
    parser.add_argument("--tfidfDim", type=int, default=0,
                         help="Concatenate a TF-IDF+truncated-SVD (LSA) feature of this many dimensions, "
                              "fit on the query corpus's OWN vocabulary, onto the dense text embedding "
                              "(dataset.TfidfTextFeaturizer). 0 (default) disables it. A corpus-native, "
                              "presence/absence-based alternative to tag_conditioning.py's external-vocab "
                              "approach -- see TfidfTextFeaturizer's docstring for why that was rejected.")
    parser.add_argument("--consistencyWeight", type=float, default=0.0,
                         help="Weight on an explicit same-font consistency loss (see diffusion.Gaussian"
                              "Diffusion.trainingLoss): penalizes the model's noise predictions for TWO "
                              "different captions of the SAME font disagreeing, under the same noise/"
                              "timestep draw. 0.0 (default) disables it entirely -- standard training only.")
    parser.add_argument("--testFraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpointDir", default=CHECKPOINT_DIR,
                         help="Defaults to the existing checkpoint directory; pass a new path when "
                              "training a variant (e.g. a different --conditioning) to compare against "
                              "it without overwriting the original.")
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

    textCenterer = None
    if args.centerText:
        textCenterer = TextCenterer.fit(trainCache)
        trainCache = textCenterer.applyToCache(trainCache)
        testCache = textCenterer.applyToCache(testCache)
        print("text queries mean-centered + renormalized (--centerText)")

    if args.tagConditioning:
        tagCache = loadTagPresenceCache()
        trainCache = concatenateTagPresence(trainCache, tagCache)
        testCache = concatenateTagPresence(testCache, tagCache)
        print("sparse tag-presence vector concatenated onto text embeddings (--tagConditioning)")

    if args.tfidfDim > 0:
        tfidfCache = loadTfidfFeatureCache(args.tfidfDim)
        trainCache = concatenateTfidfCache(trainCache, tfidfCache)
        testCache = concatenateTfidfCache(testCache, tfidfCache)
        print(f"TF-IDF+SVD ({args.tfidfDim}-dim) text features concatenated (--tfidfDim, precomputed cache -- "
              f"run build_tfidf_cache.py --dim {args.tfidfDim} first if this errors)")

    whitener = PCAWhitener.fit(fontEmbeddings, names, numComponents=args.pcaDim)

    trainSet = QueryFontDataset(trainCache, fontEmbeddings, whitener)
    trainLoader = DataLoader(trainSet, batch_size=args.batchSize, shuffle=True,
                              collate_fn=QueryFontDataset.collate, drop_last=True,
                              num_workers=args.numWorkers, persistent_workers=args.numWorkers > 0,
                              pin_memory=torch.cuda.is_available())

    testSet = QueryFontDataset(testCache, fontEmbeddings, whitener)
    testLoader = DataLoader(testSet, batch_size=args.batchSize, shuffle=False,
                             collate_fn=QueryFontDataset.collate)

    ambientDim = next(iter(fontEmbeddings.values())).shape[0]
    textDim = next(iter(trainCache.values())).shape[-1]

    if args.architecture == "shapeI":
        model = TwoPhaseDiffusionMLP(visualDim=args.pcaDim, textDim=textDim, hiddenDim=args.hiddenDim,
                                      timeDim=args.timeDim, numPlainBlocks=args.numPlainBlocks,
                                      numConditionedBlocks=args.numConditionedBlocks).to(device)
    else:
        model = DiffusionMLP(visualDim=args.pcaDim, textDim=textDim, hiddenDim=args.hiddenDim,
                              depth=args.depth, conditioning=args.conditioning).to(device)
    numParams = sum(p.numel() for p in model.parameters())
    print(f"architecture={args.architecture}  hiddenDim={args.hiddenDim}  total params={numParams:,}")
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningRate)

    os.makedirs(args.checkpointDir, exist_ok=True)

    for epoch in range(args.epochs):
        epochStart = time.time()
        model.train()
        totalLoss, steps = 0.0, 0
        for batch in trainLoader:
            text = batch["text"].to(device)
            visual = batch["visual"].to(device)
            pairedText = batch["pairedText"].to(device) if args.consistencyWeight > 0 else None

            loss = diffusion.trainingLoss(model, visual, text, pairedText=pairedText,
                                           consistencyWeight=args.consistencyWeight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
            steps += 1

        model.eval()
        testLoss, testSteps = 0.0, 0
        with torch.no_grad():
            for batch in testLoader:
                text = batch["text"].to(device)
                visual = batch["visual"].to(device)
                testLoss += diffusion.trainingLoss(model, visual, text).item()
                testSteps += 1

        print(f"epoch {epoch + 1}/{args.epochs}  trainLoss {totalLoss / max(steps, 1):.5f}  "
              f"testLoss {testLoss / max(testSteps, 1):.5f}  time {time.time() - epochStart:.2f}s")

    torch.save(model.state_dict(), os.path.join(args.checkpointDir, "checkpoint.pt"))
    whitener.save(os.path.join(args.checkpointDir, "whitener.npz"))
    if textCenterer is not None:
        textCenterer.save(os.path.join(args.checkpointDir, "textCenter.npz"))

    config = vars(args) | {"visualDim": args.pcaDim, "ambientDim": ambientDim, "textDim": textDim}
    with open(os.path.join(args.checkpointDir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # test split as (font, vector) pairs -- evaluate.py re-reads the same
    # sentence cache and pulls each vector out by index at eval time.
    testPairs = [{"font": name, "index": i} for name, vectors in testCache.items() for i in range(len(vectors))]
    with open(os.path.join(args.checkpointDir, "test_pairs.json"), "w") as f:
        json.dump(testPairs, f)

    print(f"Saved checkpoint, PCA whitener, config, and held-out test pairs to {args.checkpointDir}")


if __name__ == "__main__":
    main()
