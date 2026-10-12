"""
Trains the text-conditioned diffusion MLP to predict noise added to a
PCA-whitened, reduced-dimensional projection of the font embeddings
(see dataset.PCAWhitener), given only the paired query embedding and
timestep. Run after embed_text_local.py has produced a sentence cache.

    python3 experiments/diffusion-searches/diffusion_sampling/train.py --config configs/diffusionTrain.json
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_baseDir = _os.path.dirname(_scriptDir)
for _p in [_baseDir] + [_os.path.join(_baseDir, _d) for _d in _os.listdir(_baseDir)
                         if _os.path.isdir(_os.path.join(_baseDir, _d))]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse

from runConfig import loadRunConfig
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import (PCAWhitener, QueryFontDataset, TextCenterer, loadRaw, splitQueryCache,
                      loadTagPresenceCache, concatenateTagPresence,
                      loadTfidfFeatureCache, concatenateTfidfCache)
from diffusion import DiffusionMLP, TwoPhaseDiffusionMLP, GaussianDiffusion



def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    """Run settings are in configs/diffusionTrain.json (copy it per variant; the checkpoint's config.json records
    every key). Keys:
      sentenceModel   must match what embed_text_local.py was run with
      pcaDim          whitened PCA subspace the diffusion operates in (all.json's effective rank is ~57.6/512;
                      64 keeps a little headroom for differentiation)
      architecture    "standard" = DiffusionMLP (hiddenDim/depth/conditioning apply); "shapeI" = TwoPhaseDiffusionMLP,
                      numPlainBlocks unconditioned blocks then numConditionedBlocks additively conditioned ones
                      (conditioning/depth ignored; timeDim is shapeI only, standard hardcodes 128)
      conditioning    "concat" or "film" (Perez et al. 2018), see diffusion.ResidualBlock
      centerText      mean-center + renormalize query embeddings (dataset.TextCenterer; fixes the BGE anisotropy that
                      compresses unrelated fonts' cosine near 0.86), fit on training queries, saved to textCenter.npz
      tagConditioning concatenate a multi-hot tag-presence vector (tag_conditioning.py; needs embeddings/tagPresence.pkl)
      tfidfDim        concatenate a corpus-native TF-IDF+SVD (LSA) feature of this size (dataset.TfidfTextFeaturizer);
                      0 disables it. tfidfOnly uses it as the only text conditioning
      consistencyWeight  same-font consistency loss weight (diffusion.GaussianDiffusion.trainingLoss); 0 disables it
    Flags only change how the process runs: numWorkers (DataLoader workers; every run so far used 0) and a
    checkpointDir override."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "diffusionTrain.json"))
    parser.add_argument("--checkpointDir", default=None, help="override the config's checkpointDir")
    parser.add_argument("--numWorkers", type=int, default=0, help="DataLoader worker processes (0 = main process)")
    args = parser.parse_args()
    checkpointDir = args.checkpointDir
    del args.checkpointDir
    cfg = loadRunConfig(args.config, args)
    if checkpointDir:
        cfg.checkpointDir = checkpointDir
    return cfg


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

    if args.tfidfDim > 0 and args.tfidfOnly:
        tfidfCache = loadTfidfFeatureCache(args.tfidfDim)
        # Re-derive the SAME train/test partition splitQueryCache already produced for sentenceCache
        # (same seed, same per-font query counts -- tfidfCache is aligned 1:1 with sentenceCache), rather
        # than slicing tfidfCache directly, which would take the same leading entries for both train and
        # test and leak data between them.
        trainCache, testCache = splitQueryCache(tfidfCache, names, testFraction=args.testFraction, seed=args.seed)
        print(f"text conditioning REPLACED with TF-IDF+SVD ({args.tfidfDim}-dim) LSA features alone "
              f"(--tfidfOnly, no dense sentence embedding used at all)")
    elif args.tfidfDim > 0:
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
