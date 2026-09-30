# Quantitative test of caption GROUNDING, not phrasing quality: does a
# caption source's semantic content actually track real visual similarity,
# or is it mostly plausible-sounding invention decorrelated from how the
# fonts actually look?
#
# Core test: for a sample of font pairs, compute (a) cosine similarity of
# their captions (mean-pooled sentence-transformer embedding per font) and
# (b) cosine similarity of their real font embeddings (embeddings/all.json,
# already trusted as a faithful visual-style space -- that's the premise
# this whole project runs on). Spearman-correlate the two similarity series.
# High correlation = captions carry real transferable visual signal.
# Near-zero = caption content is decorrelated from true visual identity,
# i.e. hallucinated/generic filler dominates.
#
# Compares OLD (generate.py / Phi-4, fontQueries.json) vs NEW (generateV2.py
# / Gemma-4-E4B, fontQueriesV2.json) on the SAME set of fonts and the SAME
# sampled pairs, so the comparison isn't confounded by which fonts happen to
# be captioned by which source -- this directly tests whether the new
# captions are more or less grounded than the old ones, which is exactly
# the open question from the trainTextMLP.py regression.

import json
import random

import numpy as np
from pygtrie import CharTrie
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = "embeddings/all.json"
OLD_QUERIES_PATH = "results/fontQueries.json"
NEW_QUERIES_PATH = "results/fontQueriesV2.json"
SENTENCE_MODEL = "all-mpnet-base-v2"
NUM_FONTS_SAMPLE = 500  # -> ~125k pairs
SEED = 0


def loadFontEmbeddings(path=EMBEDDINGS_PATH):
    with open(path) as f:
        data = json.load(f)
    return {name: np.array(v, dtype=np.float32) for name, v in data.items()}


def loadQueries(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v}


def matchToEmbeddings(fontEmbeddings, queries):
    """Same longest-prefix trie match trainTextMLP.py uses -- embeddings/all.json
    is keyed by per-style-variant name, query files by base family name."""
    trie = CharTrie()
    for key in queries:
        trie[key] = key
    candidates = {}
    for embName, vector in fontEmbeddings.items():
        match = trie.longest_prefix(embName.strip())
        if match.key is None:
            continue
        candidates.setdefault(match.key, []).append((embName, vector))
    return {k: min(v, key=lambda p: len(p[0]))[1] for k, v in candidates.items()}


def meanCaptionEmbedding(model, texts):
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vecs.mean(axis=0)


def groundingCorrelation(label, fontVecByName, captionVecByName, sampleNames):
    fontMatrix = np.stack([fontVecByName[n] for n in sampleNames])
    capMatrix = np.stack([captionVecByName[n] for n in sampleNames])

    fontMatrix = fontMatrix / np.linalg.norm(fontMatrix, axis=1, keepdims=True)
    capMatrix = capMatrix / np.linalg.norm(capMatrix, axis=1, keepdims=True)

    fontSim = fontMatrix @ fontMatrix.T
    capSim = capMatrix @ capMatrix.T

    iu = np.triu_indices(len(sampleNames), k=1)
    rho, p = spearmanr(fontSim[iu], capSim[iu])
    print(f"{label}: n_fonts={len(sampleNames)} n_pairs={len(iu[0])} spearman_rho={rho:.4f} p={p:.2e}")
    return rho


def main():
    rng = random.Random(SEED)

    print("Loading font embeddings ...")
    fontEmbeddings = loadFontEmbeddings()

    print("Loading old and new captions ...")
    oldQueries = loadQueries(OLD_QUERIES_PATH)
    newQueries = loadQueries(NEW_QUERIES_PATH)

    oldMatched = matchToEmbeddings(fontEmbeddings, oldQueries)
    newMatched = matchToEmbeddings(fontEmbeddings, newQueries)

    common = sorted(set(oldMatched) & set(newMatched) & set(oldQueries) & set(newQueries))
    print(f"{len(oldMatched)} old-matched, {len(newMatched)} new-matched, "
          f"{len(common)} in BOTH (this is the controlled comparison set)")

    sampleNames = rng.sample(common, min(NUM_FONTS_SAMPLE, len(common)))

    print(f"Loading {SENTENCE_MODEL} ...")
    model = SentenceTransformer(SENTENCE_MODEL)

    print("Embedding captions (mean-pooled per font) for the sampled fonts ...")
    oldCapVecs = {n: meanCaptionEmbedding(model, oldQueries[n]) for n in sampleNames}
    newCapVecs = {n: meanCaptionEmbedding(model, newQueries[n]) for n in sampleNames}
    fontVecs = {n: oldMatched[n] for n in sampleNames}  # same underlying font embedding either way

    print()
    groundingCorrelation("OLD (Phi-4) captions   ", fontVecs, oldCapVecs, sampleNames)
    groundingCorrelation("NEW (Gemma-4-E4B) captions", fontVecs, newCapVecs, sampleNames)


if __name__ == "__main__":
    main()
