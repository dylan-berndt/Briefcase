"""
One-time precompute of the TF-IDF+SVD (LSA) per-query feature cache
(dataset.TfidfTextFeaturizer), so train.py/evaluate.py/evaluate_
branching.py just load a finished pickle instead of re-fitting AND
re-transforming on every single run.

Two real costs were happening on every training run before this existed:
(1) the TfidfVectorizer+TruncatedSVD fit itself, once, over the full
training-query corpus, and (2) a much bigger, previously-unmeasured
inefficiency: the old inline code called featurizer.transform() SEPARATELY
for each of ~17,056 fonts (a Python loop, each call transforming only
that font's ~13 queries) instead of one batched call over all ~270k
queries at once -- sklearn's per-call overhead doesn't amortize over such
tiny batches, so this was almost certainly the dominant cost, not the
one-time SVD fit. Both are gone here: fit once, transform once, save.

Fit on the FULL query corpus (not train-only) -- matches how embed_text_
local.py itself precomputes the dense sentence embeddings once,
independent of any particular train/test split; train.py's own
splitQueryCache still controls which QUERIES are used for training vs.
held-out eval, same as it already does for the dense embeddings.

    python3 experiments/diffusion-searches/build_tfidf_cache.py --dim 256
"""
import argparse
import json
import pickle

import numpy as np

from dataset import TfidfTextFeaturizer, tfidfFeatureCachePath

QUERIES_PATH = "results/fontQueriesV2.json"
GENERIC_FONTS = {"noto", "unifont", "quivira", "symbola", "dejavu", "gnu unifont"}


def loadDescriptions(path=QUERIES_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v and not any(k.lower().startswith(p) for p in GENERIC_FONTS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    args = parser.parse_args()

    descriptions = loadDescriptions()
    names = list(descriptions.keys())
    flatQueries, spans = [], []
    for name in names:
        queries = descriptions[name]
        spans.append((len(flatQueries), len(flatQueries) + len(queries)))
        flatQueries.extend(queries)
    print(f"{len(names)} fonts, {len(flatQueries)} total queries")

    print(f"fitting TF-IDF+SVD (dim={args.dim}) ...")
    featurizer = TfidfTextFeaturizer.fit(flatQueries, numComponents=args.dim)

    print("transforming all queries in ONE batched call ...")
    allVectors = featurizer.transform(flatQueries)

    cache = {name: allVectors[start:end] for name, (start, end) in zip(names, spans)}

    outPath = tfidfFeatureCachePath(args.dim)
    with open(outPath, "wb") as f:
        pickle.dump(cache, f)
    print(f"saved {len(cache)} fonts' LSA feature vectors to {outPath}")

    featurizerPath = outPath.replace("tfidfFeatures_", "tfidfFeaturizer_").replace(".pkl", "_model.pkl")
    featurizer.save(featurizerPath)
    print(f"saved fitted featurizer to {featurizerPath} (for transforming novel text later, e.g. real queries)")


if __name__ == "__main__":
    main()
