"""
The real question Flickr was brought in to answer: does VISUAL pairwise similarity correlate with
TEXT pairwise similarity at all, for random pairs of items -- independent of ASIF, independent of
any postprocessing. If it doesn't, no amount of geometric massaging of the visual space alone can
ever fix retrieval, because the visual embedding contains no signal the text side could latch onto.

For each domain: sample random pairs of items, compute visual cosine (mean-pooled per-item vector)
and text cosine (mean-pooled per-item vector, averaging multiple captions/queries per item), then
Pearson AND Spearman correlation between the two, on the SAME pairs.

    python experiments/embedding-geometry/visual_text_correlation.py
"""
import json, os, pickle
import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
rng = np.random.RandomState(0)


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def sample_pair_correlation(visual, text, n_pairs=20000, seed=0):
    """visual, text: [N, d] arrays, one row per item (already matched/aligned by index)."""
    n = len(visual)
    r = np.random.RandomState(seed)
    i = r.randint(0, n, n_pairs); j = r.randint(0, n, n_pairs)
    mask = i != j; i, j = i[mask], j[mask]
    vCos = np.sum(normalize(visual)[i] * normalize(visual)[j], axis=1)
    tCos = np.sum(normalize(text)[i] * normalize(text)[j], axis=1)
    pear = pearsonr(vCos, tCos)
    spear = spearmanr(vCos, tCos)
    return pear.statistic, spear.statistic, vCos, tCos


def flickr():
    with open(os.path.join(SCRIPT_DIR, "data", "image_embeddings.pkl"), "rb") as f:
        images = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, "data", "text_embeddings.pkl"), "rb") as f:
        texts = pickle.load(f)
    names = sorted(set(images.keys()) & set(texts.keys()))
    visual = np.stack([images[n] for n in names]).astype(np.float64)
    text = np.stack([np.mean(texts[n], axis=0) for n in names]).astype(np.float64)
    return visual, text, names


def fonts(embeddingsName="all"):
    with open(os.path.join(REPO_ROOT, "embeddings", f"{embeddingsName}.json")) as f:
        fontEmbeddings = json.load(f)
    with open(os.path.join(REPO_ROOT, "embeddings", "sentenceQueries_BAAI_bge-large-en-v1.5.pkl"), "rb") as f:
        sentenceCache = pickle.load(f)
    names = sorted(set(fontEmbeddings.keys()) & set(sentenceCache.keys()))
    visual = np.array([fontEmbeddings[n] for n in names], dtype=np.float64)
    text = np.array([np.mean(sentenceCache[n], axis=0) for n in names], dtype=np.float64)
    return visual, text, names


def report(label, visual, text):
    pear, spear, vCos, tCos = sample_pair_correlation(visual, text)
    print(f"{label:40s} n_items={len(visual):6d}  Pearson(vCos,tCos)={pear:+.4f}  Spearman={spear:+.4f}  "
          f"vCos[mean,std]=[{vCos.mean():.3f},{vCos.std():.3f}]  tCos[mean,std]=[{tCos.mean():.3f},{tCos.std():.3f}]")


if __name__ == "__main__":
    v, t, n = flickr()
    report("Flickr8k: DINOv2 image vs BGE caption", v, t)

    v, t, n = fonts("all")
    report("Fonts (best/all.json) vs BGE query", v, t)

    v, t, n = fonts("all_weak_sigreg_step85500")
    report("Fonts (weak-sigreg step85500) vs BGE query", v, t)


def allButTheTop(values, numRemove):
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[:numRemove]
    return centered - (centered @ top.T) @ top


def reportBothSidesABTT(label, visual, text, vRemove, tRemove):
    v = allButTheTop(visual, vRemove) if vRemove else visual
    t = allButTheTop(text, tRemove) if tRemove else text
    pear, spear, vCos, tCos = sample_pair_correlation(v, t)
    print(f"{label:45s} v-remove={vRemove:2d} t-remove={tRemove:2d}  Pearson={pear:+.4f}  Spearman={spear:+.4f}  "
          f"vCos[mean,std]=[{vCos.mean():.3f},{vCos.std():.3f}]  tCos[mean,std]=[{tCos.mean():.3f},{tCos.std():.3f}]")
