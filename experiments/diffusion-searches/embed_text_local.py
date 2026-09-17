"""
Text-embedding step meant to be run LOCALLY (not on a slow CPU-only
sandbox) -- lifted directly from experiments/text-searches/trainTextMLP.py's
encodeDescriptions() / matchEmbeddingsToDescriptions(), factored out as a
standalone step so the diffusion pipeline (dataset.py/train.py/evaluate.py)
can consume its cache without depending on trainTextMLP.py itself.

The only real change from trainTextMLP.py's version: SENTENCE_MODEL
defaults to a [CLS]-pooling model (BAAI/bge-large-en-v1.5) instead of
all-mpnet-base-v2 (mean pooling) -- swap via --model, the cache path
encodes the model name so switching models never silently reuses a stale
cache.

    python3 experiments/embed_text_local.py
    python3 experiments/embed_text_local.py --model BAAI/bge-base-en-v1.5
"""
import argparse
import json
import os
import pickle

import numpy as np
from pygtrie import CharTrie

EMBEDDINGS_PATH = os.path.join("embeddings", "all.json")
QUERIES_PATH = os.path.join("results", "fontQueriesV2.json")

# Same exclusion list as utils.querying.GENERIC_FONTS / trainTextMLP.py
# (copied rather than imported -- avoids dragging in spacy/cv2/transformers).
GENERIC_FONTS = {"noto", "unifont", "quivira", "symbola", "dejavu", "gnu unifont"}


def loadFontEmbeddings(path):
    with open(path, "r") as file:
        data = json.load(file)
    return {name: np.array(vector, dtype=np.float32) for name, vector in data.items()}


def loadDescriptions(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {k: v for k, v in data.items() if v and not any(k.lower().startswith(p) for p in GENERIC_FONTS)}


def matchEmbeddingsToDescriptions(fontEmbeddings, descriptions):
    """
    embeddings/all.json is keyed by per-style-variant render name;
    fontQueriesV2.json is keyed by base family name. Match with a
    longest-prefix trie, then for families with multiple matching style
    variants keep the shortest-named one as the canonical "regular"
    embedding (style suffixes only ever add characters).
    """
    trie = CharTrie()
    for key in descriptions:
        trie[key] = key

    candidates = {}
    for embName, vector in fontEmbeddings.items():
        match = trie.longest_prefix(embName.strip())
        if match.key is None:
            continue
        candidates.setdefault(match.key, []).append((embName, vector))

    return {descKey: min(options, key=lambda pair: len(pair[0]))[1] for descKey, options in candidates.items()}


def encodeDescriptions(descriptions, names, cachePath, modelName):
    if os.path.exists(cachePath):
        with open(cachePath, "rb") as file:
            cached = pickle.load(file)
        if all(name in cached for name in names):
            print(f"Using cached embeddings at {cachePath}")
            return cached

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(modelName)

    flatQueries = []
    spans = []
    for name in names:
        queries = descriptions[name]
        spans.append((len(flatQueries), len(flatQueries) + len(queries)))
        flatQueries.extend(queries)

    print(f"Embedding {len(flatQueries)} descriptions across {len(names)} fonts with {modelName} ...")
    vectors = model.encode(flatQueries, batch_size=256, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)

    encoded = {name: vectors[start:end] for name, (start, end) in zip(names, spans)}

    os.makedirs(os.path.dirname(cachePath), exist_ok=True)
    with open(cachePath, "wb") as file:
        pickle.dump(encoded, file)

    return encoded


def cachePathFor(modelName):
    safeName = modelName.replace("/", "_")
    return os.path.join("embeddings", f"sentenceQueries_{safeName}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                         help="Any sentence-transformers model name. Defaults to a [CLS]-pooling model.")
    parser.add_argument("--embeddings", default=EMBEDDINGS_PATH)
    parser.add_argument("--queries", default=QUERIES_PATH)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only embed the first N matched fonts -- for a quick smoke test.")
    args = parser.parse_args()

    print("Loading font embeddings ...")
    fontEmbeddings = loadFontEmbeddings(args.embeddings)
    print("Loading descriptions ...")
    descriptions = loadDescriptions(args.queries)

    print("Matching description families to font embeddings ...")
    matched = matchEmbeddingsToDescriptions(fontEmbeddings, descriptions)
    names = sorted(matched.keys())
    print(f"{len(names)} fonts with both a description and an embedding")
    if args.limit is not None:
        names = names[:args.limit]
        print(f"--limit set, only embedding {len(names)} fonts")

    cachePath = cachePathFor(args.model) if args.limit is None else cachePathFor(args.model) + f".limit{args.limit}"
    encodeDescriptions(descriptions, names, cachePath, args.model)
    print(f"Done. Sentence cache: {cachePath}")


if __name__ == "__main__":
    main()
