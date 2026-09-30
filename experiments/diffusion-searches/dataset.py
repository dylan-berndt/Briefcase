"""
Loads embeddings/all.json (the contrastively-finetuned ViTEmbedder font
embeddings actually used by the rest of this project's search work, see
CLAUDE.md) and a sentence-cache pickle produced by embed_text_local.py
(results/fontQueriesV2.json queries, embedded with a swappable
sentence-transformers model), and builds a train/test split that holds out
TEXT PROMPTS, not fonts -- every font that has any query at all is present
in training, and (when it has more than one query) also has held-out
queries in test. Matches experiments/text-searches/trainTextMLP.py's own
matching/splitting conventions so results are comparable.
"""
import json
import os

import numpy as np
import torch
from pygtrie import CharTrie
from torch.utils.data import Dataset

EMBEDDINGS_PATH = os.path.join("embeddings", "all.json")
QUERIES_PATH = os.path.join("results", "fontQueriesV2.json")
TAG_PRESENCE_PATH = os.path.join("embeddings", "tagPresence.pkl")

GENERIC_FONTS = {"noto", "unifont", "quivira", "symbola", "dejavu", "gnu unifont"}


def loadTagPresenceCache(path=TAG_PRESENCE_PATH):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def tfidfFeatureCachePath(dim):
    return os.path.join("embeddings", f"tfidfFeatures_d{dim}.pkl")


def loadTfidfFeatureCache(dim):
    """
    Loads the precomputed per-query LSA feature cache (build_tfidf_cache.py)
    -- same {fontName: [nQueries, dim]} shape/order convention as
    sentenceQueries_*.pkl and tagPresence.pkl. Fit ONCE, offline, on the
    full query corpus (matching how embed_text_local.py itself precomputes
    the dense sentence embeddings once, independent of any particular
    train/test split) -- avoids both the TF-IDF+SVD fit cost AND a real,
    measured inefficiency in the old inline approach (looping
    featurizer.transform() once per font instead of one batched call over
    the whole corpus) on every single training run.
    """
    import pickle
    with open(tfidfFeatureCachePath(dim), "rb") as f:
        return pickle.load(f)


def concatenateFeatureCache(sentenceCache, featureCache):
    """
    Appends each query's precomputed feature vector (tag-presence, TF-IDF/
    LSA, or anything else with this shape) onto its dense sentence
    embedding -- a pure data-level augmentation of the conditioning
    vector, no model architecture change needed (DiffusionMLP's
    textProjection already handles whatever textDim it's given). Both
    caches must share the same key set and per-font query order (true for
    any cache built from results/fontQueriesV2.json with the same
    GENERIC_FONTS filter, in the same order); fonts missing from
    featureCache (shouldn't normally happen) fall back to an all-zero
    feature vector rather than erroring.
    """
    numFeatures = next(iter(featureCache.values())).shape[1]
    combined = {}
    for name, textVectors in sentenceCache.items():
        featureVectors = featureCache.get(name)
        if featureVectors is None or featureVectors.shape[0] != textVectors.shape[0]:
            featureVectors = np.zeros((textVectors.shape[0], numFeatures), dtype=np.float32)
        combined[name] = np.concatenate([np.asarray(textVectors, dtype=np.float32), featureVectors], axis=-1)
    return combined


def concatenateTagPresence(sentenceCache, tagCache):
    return concatenateFeatureCache(sentenceCache, tagCache)


def concatenateTfidfCache(sentenceCache, tfidfCache):
    return concatenateFeatureCache(sentenceCache, tfidfCache)


def loadFontEmbeddings(path=EMBEDDINGS_PATH):
    with open(path, "r") as file:
        data = json.load(file)
    return {name: np.array(vector, dtype=np.float32) for name, vector in data.items()}


def loadDescriptions(path=QUERIES_PATH):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {k: v for k, v in data.items() if v and not any(k.lower().startswith(p) for p in GENERIC_FONTS)}


def matchEmbeddingsToDescriptions(fontEmbeddings, descriptions):
    """Same canonical (shortest-name-per-family) matching trainTextMLP.py uses."""
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


def loadRaw(embeddingsPath=EMBEDDINGS_PATH, queriesPath=QUERIES_PATH, sentenceCachePath=None):
    import pickle

    fontEmbeddings = loadFontEmbeddings(embeddingsPath)
    descriptions = loadDescriptions(queriesPath)
    matched = matchEmbeddingsToDescriptions(fontEmbeddings, descriptions)

    with open(sentenceCachePath, "rb") as file:
        sentenceCache = pickle.load(file)

    names = sorted(name for name in matched if name in sentenceCache)
    matched = {name: matched[name] for name in names}
    sentenceCache = {name: sentenceCache[name] for name in names}
    return matched, sentenceCache


def splitQueryCache(sentenceCache, names, testFraction=0.2, seed=1234):
    """
    Per-font split of a font's own cached query embeddings into train/test
    halves -- not a font-level split (every font appears in both, when it
    has more than one query), matching trainTextMLP.py's splitQueryCache.
    """
    rng = np.random.default_rng(seed)
    trainCache, testCache = {}, {}
    for name in names:
        vectors = sentenceCache[name]
        order = rng.permutation(len(vectors))
        testCount = min(max(1, round(len(vectors) * testFraction)), len(vectors) - 1) if len(vectors) > 1 else 0
        testCache[name] = vectors[order[:testCount]]
        trainCache[name] = vectors[order[testCount:]]
    return trainCache, testCache


class PCAWhitener:
    """
    Projects the 512-d ambient font-embedding space down to a lower-
    dimensional PCA subspace and whitens it (unit variance per component),
    so the diffusion process operates where the data's real variance
    actually lives instead of spending capacity on the ~450+ near-
    degenerate ambient dimensions (effective_dimension.py measured
    embeddings/all.json's participation ratio at ~57.6 out of 512) that
    would otherwise just contribute accumulated noise to every sample.
    Same idea utils/search.py's GridFeedbackSearch already uses (centre ->
    PCA -> whiten) for the same anisotropic embedding space, applied here
    to the diffusion target instead of a search index. transform/
    inverseTransform work on both a single [ambientDim] vector and a
    batch [N, ambientDim].
    """

    def __init__(self, mean, components, componentStd):
        self.mean = mean                  # [ambientDim]
        self.components = components      # [numComponents, ambientDim], PCA directions (rows)
        self.componentStd = componentStd  # [numComponents], sqrt of each component's eigenvalue

    @staticmethod
    def fit(fontEmbeddings, fontKeys, numComponents=64):
        from sklearn.decomposition import PCA

        values = np.stack([fontEmbeddings[k] for k in fontKeys], axis=0).astype(np.float64)
        mean = values.mean(axis=0)

        pca = PCA(n_components=numComponents)
        pca.fit(values - mean)
        componentStd = np.sqrt(pca.explained_variance_) + 1e-6

        return PCAWhitener(mean.astype(np.float32), pca.components_.astype(np.float32),
                            componentStd.astype(np.float32))

    def transform(self, x):
        return ((x - self.mean) @ self.components.T) / self.componentStd

    def inverseTransform(self, z):
        return (z * self.componentStd) @ self.components + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, components=self.components, componentStd=self.componentStd)

    @staticmethod
    def load(path):
        data = np.load(path)
        return PCAWhitener(data["mean"], data["components"], data["componentStd"])


class TextCenterer:
    """
    Mean-centers text query embeddings, then renormalizes to unit norm.
    Fixes anisotropy measured directly in the BGE-large query embedding
    space used here: mean-vector norm 0.928 (of a unit-norm space), only
    ~21.5/1024 effective dimensions -- a dominant "generic" direction
    that compresses raw cosine similarity between UNRELATED fonts' text
    up near 0.86, masking the real, weaker discriminative signal
    (data_quality_check.py: corr(text sim, visual distance) roughly
    doubles in magnitude, -0.065 -> -0.110, after this transform). Same
    "All-but-the-Top" style fix (Mu & Viswanath) already used elsewhere
    in this project for the visual embedding space, applied here to the
    text side for the first time. Fit on TRAINING queries only and
    reused as-is on test/inference queries, matching PCAWhitener's own
    train/apply convention.
    """

    def __init__(self, mean):
        self.mean = mean  # [textDim]

    @staticmethod
    def fit(queryCache):
        allVectors = np.concatenate([np.asarray(v, dtype=np.float64) for v in queryCache.values()], axis=0)
        return TextCenterer(allVectors.mean(axis=0).astype(np.float32))

    def transform(self, x):
        v = np.asarray(x, dtype=np.float32) - self.mean
        norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
        return v / norm

    def applyToCache(self, queryCache):
        return {name: self.transform(vectors) for name, vectors in queryCache.items()}

    def save(self, path):
        np.savez(path, mean=self.mean)

    @staticmethod
    def load(path):
        data = np.load(path)
        return TextCenterer(data["mean"])


class TfidfTextFeaturizer:
    """
    TF-IDF + truncated SVD (i.e. Latent Semantic Analysis) over the query
    corpus's OWN vocabulary -- an explicit, presence/absence-based,
    corpus-native complement to the dense pretrained sentence embedding.
    Motivation: a single caption mentions only a partial, noisy SUBSET of
    a font's true style attributes; different captions of the same font
    surface different subsets (the measured dominant noise source, not
    paraphrase variation). TF-IDF is inherently a bag-of-words presence/
    absence representation, and SVD's low-rank structure pulls
    synonymous/co-occurring words together using THIS corpus's own
    statistics.

    A first attempt at synonym-robustness used spaCy word-vector
    similarity against the retrieval-head's external 2280-tag vocabulary
    (tag_conditioning.py) and was rejected after direct measurement: a
    0.5 cosine threshold against that vocabulary matched an average of 64
    tags per query, almost all unrelated to the query's actual content
    (e.g. "flowing cursive handwriting with elegant swashes" matched
    "financial", "government", "nautical", "political", "vacation") --
    far too permissive to use as conditioning. TF-IDF+SVD avoids that
    failure mode entirely: it never needs an external vocabulary or a
    similarity threshold, and IDF naturally downweights generic filler
    words (no manual stopword list needed either).

    Fit on TRAINING queries' raw text only (both the vectorizer's IDF
    statistics and the SVD components), applied as-is to test/inference
    queries -- same train/apply convention as PCAWhitener/TextCenterer.
    """

    def __init__(self, vectorizer, svd):
        self.vectorizer = vectorizer
        self.svd = svd

    @staticmethod
    def fit(texts, numComponents=100):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        tfidf = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=numComponents, random_state=0)
        svd.fit(tfidf)
        return TfidfTextFeaturizer(vectorizer, svd)

    def transform(self, texts):
        return self.svd.transform(self.vectorizer.transform(texts)).astype(np.float32)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        return TfidfTextFeaturizer(data["vectorizer"], data["svd"])


def concatenateTfidfFeatures(sentenceCache, rawTextCache, featurizer):
    """Appends each query's TF-IDF+SVD feature vector onto its dense sentence embedding."""
    numComponents = featurizer.svd.n_components
    combined = {}
    for name, textVectors in sentenceCache.items():
        rawTexts = rawTextCache.get(name)
        if rawTexts is None or len(rawTexts) != textVectors.shape[0]:
            tfidfVecs = np.zeros((textVectors.shape[0], numComponents), dtype=np.float32)
        else:
            tfidfVecs = featurizer.transform(list(rawTexts))
        combined[name] = np.concatenate([np.asarray(textVectors, dtype=np.float32), tfidfVecs], axis=-1)
    return combined


class QueryFontDataset(Dataset):
    """
    One item = one (query embedding, PCA-whitened visual embedding) pair,
    drawn from a font's train- or test-half query cache. Also returns
    "pairedText": a DIFFERENT query of the SAME font (drawn fresh each
    __getitem__ call) when the font has more than one query available in
    this cache, else the same text repeated (a harmless no-op for the
    consistency loss -- see diffusion.GaussianDiffusion.trainingLoss's
    consistencyWeight). This is what lets training explicitly teach "two
    different captions of one font are noisy partial views of the same
    target," rather than relying only on the implicit consistency that
    sharing a regression target already gives for free.
    """

    def __init__(self, queryCache, fontEmbeddings, whitener: PCAWhitener):
        self.fontEmbeddings = fontEmbeddings
        self.whitener = whitener
        self.index = [(name, i) for name, vectors in queryCache.items() for i in range(len(vectors))]
        self.queryCache = queryCache

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        name, vectorIndex = self.index[i]
        vectors = self.queryCache[name]
        text = vectors[vectorIndex]
        if len(vectors) > 1:
            otherIndex = vectorIndex
            while otherIndex == vectorIndex:
                otherIndex = np.random.randint(len(vectors))
            pairedText = vectors[otherIndex]
        else:
            pairedText = text
        visual = self.whitener.transform(self.fontEmbeddings[name]).astype(np.float32)
        return {
            "text": torch.from_numpy(np.asarray(text, dtype=np.float32)),
            "pairedText": torch.from_numpy(np.asarray(pairedText, dtype=np.float32)),
            "visual": torch.from_numpy(visual),
            "font": name,
        }

    @staticmethod
    def collate(samples):
        return {
            "text": torch.stack([s["text"] for s in samples]),
            "pairedText": torch.stack([s["pairedText"] for s in samples]),
            "visual": torch.stack([s["visual"] for s in samples]),
            "font": [s["font"] for s in samples],
        }
