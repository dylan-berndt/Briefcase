"""
Loads the precomputed visual (font) and text (query) embeddings produced by
embed_fonts.py / embed_text.py, and builds a train/test split that holds out
TEXT PROMPTS, not fonts -- every font that has any query at all is present in
training, and (when it has more than one query) also has held-out queries in
test. This matches the experiment's goal: measure whether text can predict a
font's visual embedding, not whether unseen fonts can be described.
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

EMBEDDINGS_DIR = "embeddings"


def loadRaw(embeddingsDir=EMBEDDINGS_DIR):
    with open(os.path.join(embeddingsDir, "font_vit_embeddings.json")) as f:
        fontEmbeddings = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
    with open(os.path.join(embeddingsDir, "text_query_embeddings.json")) as f:
        textEmbeddings = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
    with open(os.path.join(embeddingsDir, "query_font_pairs.json")) as f:
        pairs = json.load(f)
    # Only keep pairs whose font actually has a visual embedding (matching can
    # produce query font-keys with no successfully-embedded font, e.g. fonts
    # missing a full a-z glyph set).
    pairs = [p for p in pairs if p["font"] in fontEmbeddings]
    return fontEmbeddings, textEmbeddings, pairs


def splitPairs(pairs, testFraction=0.2, seed=1234, minTestQueries=1):
    """
    Groups (font, query) pairs by font and splits each font's queries into
    train/test independently, so every font with >= 2 queries appears on
    both sides and no font is fully held out.
    """
    rng = random.Random(seed)
    byFont = {}
    for p in pairs:
        byFont.setdefault(p["font"], []).append(p)

    train, test = [], []
    for font, fontPairs in byFont.items():
        fontPairs = fontPairs[:]
        rng.shuffle(fontPairs)
        nTest = int(round(len(fontPairs) * testFraction))
        nTest = min(nTest, len(fontPairs) - 1) if len(fontPairs) > 1 else 0
        nTest = max(nTest, minTestQueries) if len(fontPairs) > minTestQueries else nTest
        test.extend(fontPairs[:nTest])
        train.extend(fontPairs[nTest:])

    return train, test


class EmbeddingStats:
    """Per-dimension mean/std of the visual embedding space the diffusion
    process operates in, so x_T ~ N(0, I) is a reasonable prior."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @staticmethod
    def fit(fontEmbeddings, fontKeys):
        values = np.stack([fontEmbeddings[k] for k in fontKeys], axis=0)
        mean = values.mean(axis=0)
        std = values.std(axis=0) + 1e-6
        return EmbeddingStats(mean.astype(np.float32), std.astype(np.float32))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path):
        data = np.load(path)
        return EmbeddingStats(data["mean"], data["std"])


class QueryFontDataset(Dataset):
    """One item = one (text embedding, normalized visual embedding) pair."""

    def __init__(self, pairs, fontEmbeddings, textEmbeddings, stats: EmbeddingStats):
        self.pairs = pairs
        self.fontEmbeddings = fontEmbeddings
        self.textEmbeddings = textEmbeddings
        self.stats = stats

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        text = self.textEmbeddings[pair["query"]]
        visual = self.stats.normalize(self.fontEmbeddings[pair["font"]])
        return {
            "text": torch.from_numpy(text),
            "visual": torch.from_numpy(visual),
            "font": pair["font"],
            "query": pair["query"],
        }

    @staticmethod
    def collate(samples):
        return {
            "text": torch.stack([s["text"] for s in samples]),
            "visual": torch.stack([s["visual"] for s in samples]),
            "font": [s["font"] for s in samples],
            "query": [s["query"] for s in samples],
        }
