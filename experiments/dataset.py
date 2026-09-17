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

GENERIC_FONTS = {"noto", "unifont", "quivira", "symbola", "dejavu", "gnu unifont"}


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


class QueryFontDataset(Dataset):
    """One item = one (query embedding, PCA-whitened visual embedding) pair,
    drawn from a font's train- or test-half query cache."""

    def __init__(self, queryCache, fontEmbeddings, whitener: PCAWhitener):
        self.fontEmbeddings = fontEmbeddings
        self.whitener = whitener
        self.index = [(name, i) for name, vectors in queryCache.items() for i in range(len(vectors))]
        self.queryCache = queryCache

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        name, vectorIndex = self.index[i]
        text = self.queryCache[name][vectorIndex]
        visual = self.whitener.transform(self.fontEmbeddings[name]).astype(np.float32)
        return {
            "text": torch.from_numpy(np.asarray(text, dtype=np.float32)),
            "visual": torch.from_numpy(visual),
            "font": name,
        }

    @staticmethod
    def collate(samples):
        return {
            "text": torch.stack([s["text"] for s in samples]),
            "visual": torch.stack([s["visual"] for s in samples]),
            "font": [s["font"] for s in samples],
        }
