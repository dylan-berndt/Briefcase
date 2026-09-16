"""
Embeds the pregenerated font-search queries (results/fontQueries.json +
results/shortQueries.json) with a sentence-transformer that pools via the
[CLS] token (BAAI/bge-base-en-v1.5), and matches each query's font name to
the font-name keys produced by embed_fonts.py (which are "{family} {style}"
combos, e.g. "Open Sans Regular", not the bare family/slug names used in
the query files). Matching reuses the same longest-prefix-match trie trick
CombinedQueryData uses in utils/querying.py.

Output:
  embeddings/text_query_embeddings.json   {query_string: [floats]}
  embeddings/query_font_pairs.json        [{"font": ..., "query": ..., "source": ...}, ...]
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygtrie import CharTrie
from sentence_transformers import SentenceTransformer

TEXT_MODEL = "BAAI/bge-large-en-v1.5"
FONT_EMBEDDINGS_PATH = os.path.join("embeddings", "font_vit_embeddings.json")
QUERY_SOURCES = [
    (os.path.join("results", "fontQueries.json"), "fontQueries"),
    (os.path.join("results", "shortQueries.json"), "shortQueries"),
]
OUTPUT_DIR = "embeddings"


def loadQueries():
    combined = {}
    perSource = {}
    for path, source in QUERY_SOURCES:
        if not os.path.exists(path):
            print(f"Missing {path}, skipping")
            continue
        with open(path, "r") as file:
            data = json.load(file)
        perSource[source] = data
        for name, queries in data.items():
            combined.setdefault(name, []).extend((q, source) for q in queries)
    return combined


def main():
    if not os.path.exists(FONT_EMBEDDINGS_PATH):
        raise FileNotFoundError(f"{FONT_EMBEDDINGS_PATH} not found -- run embed_fonts.py first")

    with open(FONT_EMBEDDINGS_PATH, "r") as file:
        fontNames = list(json.load(file).keys())

    combined = loadQueries()
    print(f"{len(combined)} distinct font keys across query files")

    trie = CharTrie()
    for key in combined:
        trie[key] = key

    pairs = []
    matched = 0
    for fontName in fontNames:
        match = trie.longest_prefix(fontName)
        if match.key is None:
            continue
        matched += 1
        for query, source in combined[match.key]:
            pairs.append({"font": fontName, "query": query, "source": source})

    print(f"{matched}/{len(fontNames)} visual fonts matched to a query entry, {len(pairs)} (font, query) pairs")

    uniqueQueries = sorted({p["query"] for p in pairs})
    print(f"Embedding {len(uniqueQueries)} unique queries with {TEXT_MODEL}")

    model = SentenceTransformer(TEXT_MODEL)
    vectors = model.encode(uniqueQueries, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    textEmbeddings = {q: vectors[i].tolist() for i, q in enumerate(uniqueQueries)}
    with open(os.path.join(OUTPUT_DIR, "text_query_embeddings.json"), "w") as file:
        json.dump(textEmbeddings, file)

    with open(os.path.join(OUTPUT_DIR, "query_font_pairs.json"), "w") as file:
        json.dump(pairs, file)

    print("Done.")


if __name__ == "__main__":
    main()
