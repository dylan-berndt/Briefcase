"""
Recall@k near random chance is consistent with two very different failure
modes: (a) the task is genuinely this hard here (plausible -- see CLAUDE.md's
evalTextNarrowing finding of R@1~=0% even for the actual contrastively-
finetuned retrieval model), or (b) the diffusion model has learned the
*marginal* shape of the visual-embedding manifold but isn't actually using
the text conditioning to pick *where* in it to land for a given query
("conditioning collapse" -- plausible independently, since minimizing
average denoising MSE doesn't require using the condition at all if the
condition's effect on the target is weak relative to how concentrated the
marginal distribution already is).

This tries to tell those apart cheaply: for a sample of held-out queries
from DISTINCT fonts, generate one sample each, then compare the pairwise
cosine-similarity structure of the GENERATED embeddings against the
pairwise cosine-similarity structure of their TRUE target fonts.

  - If conditioning is working: generated-pair similarities should
    correlate with true-pair similarities (queries whose true fonts are
    alike should produce alike outputs, and vice versa).
  - If conditioning has collapsed: generated embeddings will look similar
    to EACH OTHER regardless of query (high mean off-diagonal similarity,
    low variance across pairs) and uncorrelated with the true structure.

Also reports each sample's cosine similarity to its own true target
(continuous quality signal, less noisy than discrete recall@k on a small
sample) and to the corpus-wide mean font embedding (a high value there for
every query, regardless of what the query was, is the direct fingerprint
of mode collapse toward "the average font").

    python3 experiments/diagnose_conditioning.py --numQueries 200
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from dataset import EmbeddingStats, loadFontEmbeddings, EMBEDDINGS_PATH
from diffusion import DiffusionMLP, GaussianDiffusion

CHECKPOINT_DIR = os.path.join("checkpoints", "diffusion")


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numQueries", type=int, default=200,
                         help="Distinct-font held-out queries to sample -- kept small since this is O(n^2) pairs.")
    parser.add_argument("--evalSteps", type=int, default=50)
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

    stats = EmbeddingStats.load(os.path.join(CHECKPOINT_DIR, "stats.npz"))
    fontEmbeddings = loadFontEmbeddings(EMBEDDINGS_PATH)
    with open(cachePathFor(config["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)

    testPairs = [p for p in testPairs if p["font"] in fontEmbeddings]

    # One query per DISTINCT font, so the "true similarity structure" being
    # checked is font-to-font, not diluted by several queries of one font.
    byFont = {}
    for p in testPairs:
        byFont.setdefault(p["font"], p)
    distinctFontPairs = list(byFont.values())
    rng.shuffle(distinctFontPairs)
    batch = distinctFontPairs[:args.numQueries]
    print(f"{len(batch)} queries, one per distinct font")

    model = DiffusionMLP(visualDim=config["visualDim"], textDim=config["textDim"],
                          hiddenDim=config["hiddenDim"], depth=config["depth"]).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "checkpoint.pt"), map_location=device))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=config["timesteps"], device=device)

    texts = torch.stack([
        torch.from_numpy(np.asarray(sentenceCache[p["font"]][p["index"]], dtype=np.float32))
        for p in batch
    ]).to(device)

    generated = diffusion.sample(model, texts, config["visualDim"], device=device, numSteps=args.evalSteps)
    generated = generated.cpu().numpy() * stats.std + stats.mean
    generated = generated / (np.linalg.norm(generated, axis=-1, keepdims=True) + 1e-8)

    trueTargets = np.stack([fontEmbeddings[p["font"]] for p in batch])
    trueTargets = trueTargets / (np.linalg.norm(trueTargets, axis=-1, keepdims=True) + 1e-8)

    allFontMatrix = np.stack(list(fontEmbeddings.values()))
    globalMean = allFontMatrix.mean(axis=0)
    globalMean = globalMean / (np.linalg.norm(globalMean) + 1e-8)

    genSim = generated @ generated.T
    trueSim = trueTargets @ trueTargets.T
    n = len(batch)
    iu = np.triu_indices(n, k=1)

    correlation = np.corrcoef(genSim[iu], trueSim[iu])[0, 1]
    ownTargetCos = np.sum(generated * trueTargets, axis=1)
    meanCos = generated @ globalMean

    print(f"\nPairwise similarity structure ({n*(n-1)//2} pairs among {n} distinct-font queries):")
    print(f"  correlation(generated-pair similarity, true-pair similarity): {correlation:.4f}")
    print(f"  mean off-diagonal generated-generated similarity: {genSim[iu].mean():.4f} (std {genSim[iu].std():.4f})")
    print(f"  mean off-diagonal true-true similarity:           {trueSim[iu].mean():.4f} (std {trueSim[iu].std():.4f})")
    print(f"\nPer-query signal:")
    print(f"  mean cosine(generated, own true target):  {ownTargetCos.mean():.4f} (std {ownTargetCos.std():.4f})")
    print(f"  mean cosine(generated, corpus-wide mean font): {meanCos.mean():.4f} (std {meanCos.std():.4f})")
    print(f"  [for reference] mean cosine(true target, corpus-wide mean font): "
          f"{(trueTargets @ globalMean).mean():.4f} (std {(trueTargets @ globalMean).std():.4f})")

    print("\nReading these together:")
    print("  - low correlation + high/tight off-diagonal generated similarity + high cosine-to-global-mean")
    print("    => conditioning collapse: the model ignores the query and outputs ~the average font regardless.")
    print("  - positive correlation, generated off-diagonal similarity closer to true's spread")
    print("    => conditioning is doing real work; near-chance recall@k likely reflects task difficulty")
    print("    (many legitimately similar fonts per style region) rather than a broken model.")


if __name__ == "__main__":
    main()
