"""
One-time precompute of embeddings/tagPresence.pkl: {fontName: [nQueries,
numTags] float32 signed presence}, in the SAME key set and per-font query
order as results/fontQueriesV2.json itself (matching dataset.py/embed_text_
local.py's own GENERIC_FONTS filtering and iteration order), so it lines
up index-for-index with any sentenceQueries_<model>.pkl cache built from
the same source file -- see tag_conditioning.py for what/why (lemma +
substring + spaCy word-vector synonym matching, not literal substring
alone).

    python3 experiments/diffusion-searches/build_tag_presence_cache.py
"""
import json
import pickle

import numpy as np

from tag_conditioning import TagMatcher

QUERIES_PATH = "results/fontQueriesV2.json"
OUT_PATH = "embeddings/tagPresence.pkl"
GENERIC_FONTS = {"noto", "unifont", "quivira", "symbola", "dejavu", "gnu unifont"}


def loadDescriptions(path=QUERIES_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v and not any(k.lower().startswith(p) for p in GENERIC_FONTS)}


def main():
    descriptions = loadDescriptions()
    matcher = TagMatcher()
    print(f"{len(descriptions)} fonts, {matcher.numTags} tags in vocab "
          f"({len(matcher.validIdx)} after stopword filtering), "
          f"synonym vectors {'enabled' if matcher.tagVectors is not None else 'DISABLED'}")

    names = list(descriptions.keys())
    flatQueries, spans = [], []
    for name in names:
        queries = descriptions[name]
        spans.append((len(flatQueries), len(flatQueries) + len(queries)))
        flatQueries.extend(queries)
    print(f"{len(flatQueries)} total queries -- running spaCy pipe (this takes a while)...")

    vectors = []
    for i, v in enumerate(matcher.presenceVectors(flatQueries, batchSize=512)):
        vectors.append(v)
        if i % 5000 == 0:
            print(f"\r{i}/{len(flatQueries)}", end="")
    print()

    cache = {name: np.stack(vectors[start:end]) for name, (start, end) in zip(names, spans)}

    allVecs = np.concatenate(list(cache.values()), axis=0)
    avgTagsHit = (allVecs > 0).sum(axis=1).mean()
    avgNegated = (allVecs < 0).sum(axis=1).mean()
    print(f"mean tags matched (positive) per query: {avgTagsHit:.2f}, mean negated per query: {avgNegated:.3f}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"saved {len(cache)} fonts' tag-presence vectors to {OUT_PATH}")


if __name__ == "__main__":
    main()
