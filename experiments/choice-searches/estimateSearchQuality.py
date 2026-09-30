# Estimates what a real user would actually see if free-text search were
# served by the newly trained embedding->TF-IDF retrieval head, using the
# EXACT SAME query-parsing mechanism as the deployed TagSearch
# (utils/search.py): spaCy lemma/substring/synonym-vector matching of query
# terms onto the vocabulary, then ranking fonts by dot product with each
# font's predicted tag vector. Reimplemented standalone here (rather than
# instantiating TagSearch) because TagSearch's __init__ pulls in the ViT
# backbone + image dataset, which this embedding-only model doesn't need.
#
# This is a QUALITATIVE look at actual results over the FULL corpus, as a
# real deployment would serve -- the quantitative generalization numbers
# are retrieval.py's P@5/P@10/R@10/median-rank on the held-out split; this
# script is for seeing what those numbers actually look like as ranked
# search results, good and bad.

import json
import os
import re

import numpy as np
import spacy
import torch
import torch.nn.functional as F

from retrieval import RetrievalHead, HIDDEN_DIM, CHECKPOINT_DIR, loadFontEmbeddings
from utils.search import NEGATIONS

SPACY_MODEL = "en_core_web_lg"
SYNONYM_THRESHOLD = 0.5
SYNONYM_SCALE = 1.0

# Meta-words that describe "this is a font search," not what STYLE of font
# is wanted -- every query is implicitly about fonts, so matching on these
# never discriminates between fonts and only adds noise (observed: "font"
# matched "webfont"/"brush-font" on literally every query in the first
# pass). Filtered before matching, not just down-weighted, since they carry
# zero style signal.
QUERY_STOPWORDS = {"font", "fonts", "typeface", "typefaces"}

QUERIES = [
    "elegant wedding script",
    "bold geometric sans for a tech logo",
    "vintage western poster font",
    "playful rounded font for kids",
    "clean minimal sans serif",
    "spooky halloween horror font",
    "futuristic sci-fi display font",
    "elegant luxury brand serif",
    "handwritten casual script",
    "monospace coding font",
    "old newspaper typewriter font",
    "art deco display font",
    "cute bubbly font",
    "grunge distressed font",
    "corporate professional sans",
    "condensed narrow display font",
    "thin light modern sans",
    "comic book font",
    "military stencil font",
    "chalkboard hand drawn font",
    "elegant italic calligraphy",
    "graffiti street font",
    "medieval gothic blackletter font",
    "not bold thin font",
    "psychedelic 70s groovy font",
    "children's storybook font",
    "sporty athletic font",
    "luxury perfume brand font",
    "space rocket sci-fi font",
    "japanese inspired brush font",
]

TOP_K = 10


def loadModel():
    with open(os.path.join(CHECKPOINT_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    with open(os.path.join(CHECKPOINT_DIR, "idf.json")) as f:
        idfByTag = json.load(f)
    idf = np.array([idfByTag[tag] for tag in vocab], dtype=np.float32)

    fontEmbeddings = loadFontEmbeddings()
    fontDim = len(next(iter(fontEmbeddings.values())))
    model = RetrievalHead(fontDim, HIDDEN_DIM, len(vocab))
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "head.pt"), map_location="cpu"))
    model.eval()
    return model, vocab, idf, fontEmbeddings


@torch.no_grad()
def predictAllFonts(model, fontEmbeddings):
    names = list(fontEmbeddings.keys())
    matrix = torch.tensor(np.stack([fontEmbeddings[n] for n in names]), dtype=torch.float32)
    preds = F.normalize(model(matrix), dim=-1).numpy()
    return names, preds


def sparsifyByConfidence(preds, idf, threshold):
    """
    Fixed-confidence threshold, not top-K: keep whichever of a font's
    normalized predicted dimensions exceed `threshold` (could be 0, could
    be 50 -- adapts per font instead of forcing every font to keep the same
    count regardless of how confident the model actually is), then rescale
    the kept ones by idf, same construction as the true training targets
    (binary presence x idf). At threshold=0.15 the median kept count across
    the corpus is ~7, matching the true training-data median tags/font.
    """
    mask = preds > threshold
    sparse = np.where(mask, idf[np.newaxis, :], 0.0).astype(np.float32)
    return sparse


def buildNLP(vocab):
    nlp = spacy.load(SPACY_MODEL)
    lemmaToTags = {}
    vectors = []
    hasVectors = nlp.vocab.vectors_length > 0
    for i, tag in enumerate(vocab):
        doc = nlp(tag.lower())
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            lemmaToTags.setdefault(token.lemma_, []).append(i)
        if hasVectors:
            norm = doc.vector_norm
            vectors.append(doc.vector / norm if norm else np.zeros_like(doc.vector))
    tagVectors = np.stack(vectors).astype(np.float32) if hasVectors else None
    return nlp, lemmaToTags, tagVectors


def _wordBoundaryContains(needle, haystack):
    """True if `needle` appears in `haystack` as a delimited component (start/
    end of string or a non-alphanumeric char on both sides) -- NOT wherever it
    happens to appear as raw characters. Fixes the observed bug where query
    word "font" substring-matched "webfont" (no real boundary between "web"
    and "font" -- it's one fused word) on every single query, while still
    allowing legitimate boundary-separated matches like "font" in "brush-font"."""
    if not needle:
        return False
    return re.search(rf'(?:^|[^a-z0-9]){re.escape(needle)}(?:$|[^a-z0-9])', haystack) is not None


def termMatches(token, vocab, lemmaToTags, tagVectors):
    """Same priority order as TagSearch._termMatches: exact lemma (1.0) and
    word-boundary substring (0.5) first; vector-similar synonyms only as a
    fallback when neither direct tier hits."""
    matches = {}
    lemma, text = token.lemma_, token.text
    for i in lemmaToTags.get(lemma, []):
        matches[i] = max(matches.get(i, 0.0), 1.0)
    for i, tag in enumerate(vocab):
        tagLower = tag.lower()
        if _wordBoundaryContains(text, tagLower) or _wordBoundaryContains(lemma, tagLower):
            matches[i] = max(matches.get(i, 0.0), 0.5)
    if matches:
        return matches
    if tagVectors is not None and token.has_vector and token.vector_norm:
        sims = tagVectors @ (token.vector / token.vector_norm)
        for i in np.where(sims >= SYNONYM_THRESHOLD)[0]:
            matches[int(i)] = float(sims[i]) * SYNONYM_SCALE
    return matches


def queryWeights(query, nlp, vocab, idf, lemmaToTags, tagVectors):
    pos = np.zeros(len(vocab), dtype=np.float32)
    neg = np.zeros(len(vocab), dtype=np.float32)
    matched, unmatched = 0, []
    negate = False
    for token in nlp(query.lower()):
        if token.is_punct or token.is_space:
            negate = False
            continue
        if token.lower_ in NEGATIONS or token.dep_ == "neg" or token.lower_ == "n't":
            negate = True
            continue
        if token.is_stop or token.lemma_ in QUERY_STOPWORDS or token.lower_ in QUERY_STOPWORDS:
            continue
        hits = termMatches(token, vocab, lemmaToTags, tagVectors)
        if hits:
            matched += 1
            target = neg if negate else pos
            for i, magnitude in hits.items():
                target[i] = max(target[i], magnitude)
        else:
            unmatched.append(token.text)
        negate = False

    # IDF-weight the query side too (see retrieval.py::buildTfidfTargets for
    # the matching font-side weighting) -- without this, a flat 1.0 match
    # weight lets a common attribute contribute to the score exactly as
    # much as a rare, actually-discriminative one, which is the direct
    # mechanism behind generic bold/italic-heavy families flooding
    # unrelated queries. Raw idf overcorrects, though: it can collapse
    # almost all of a multi-term query's weight onto a single rare
    # dimension, making the ranking fully dependent on that one dimension's
    # prediction quality (observed regression: "vintage western poster"
    # got WORSE once "western" alone dominated over "poster"+"vintage").
    # log1p(idf) is the sublinear fix -- idf is always >= 0 (log(N/docFreq)
    # with docFreq <= N), so plain log(idf) would hit log(0) for the most
    # common terms; log1p keeps common terms at exactly 0 like before while
    # compressing how far rare terms can pull ahead of everything else.
    sublinearIdf = np.log1p(idf)
    return (pos - neg) * sublinearIdf, matched, unmatched


CONFIDENCE_THRESHOLD = 0.15


def main():
    print("Loading model + vocab + font embeddings ...")
    model, vocab, idf, fontEmbeddings = loadModel()
    print(f"{len(vocab)} vocab terms, {len(fontEmbeddings)} fonts\n")

    print("Predicting per-font vectors for the full corpus ...")
    names, preds = predictAllFonts(model, fontEmbeddings)

    sparsePreds = sparsifyByConfidence(preds, idf, CONFIDENCE_THRESHOLD)
    keptCounts = (sparsePreds != 0).sum(axis=1)
    sparseNorms = np.linalg.norm(sparsePreds, axis=1, keepdims=True)
    sparseNorms[sparseNorms == 0] = 1.0
    sparsePreds = sparsePreds / sparseNorms
    print(f"Confidence threshold {CONFIDENCE_THRESHOLD}: median {int(np.median(keptCounts))} "
          f"kept attributes/font (adapts per font), {int((keptCounts == 0).sum())} fonts with none kept\n")

    print("Loading spaCy + building tag vectors ...")
    nlp, lemmaToTags, tagVectors = buildNLP(vocab)

    for query in QUERIES:
        weights, matched, unmatched = queryWeights(query, nlp, vocab, idf, lemmaToTags, tagVectors)

        nonzero = sorted(np.nonzero(weights)[0], key=lambda i: -abs(weights[i]))
        matchedTerms = [(vocab[i], round(float(weights[i]), 2)) for i in nonzero[:15]]

        print(f"\n=== '{query}' ===")
        print(f"  matched {matched} query word(s) -> vocab terms: {matchedTerms}")
        if unmatched:
            print(f"  unmatched query words: {unmatched}")

        denseScores = preds @ weights
        sparseScores = sparsePreds @ weights

        print("  -- dense (raw model output) --")
        for rank, idx in enumerate(np.argsort(-denseScores)[:TOP_K], 1):
            print(f"    {rank:>2}. {names[idx]:<40} score={denseScores[idx]:.3f}")
        print(f"  -- confidence-thresholded (>{CONFIDENCE_THRESHOLD}) --")
        for rank, idx in enumerate(np.argsort(-sparseScores)[:TOP_K], 1):
            print(f"    {rank:>2}. {names[idx]:<40} score={sparseScores[idx]:.3f}")


if __name__ == "__main__":
    main()
