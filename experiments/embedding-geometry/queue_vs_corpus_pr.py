"""
Small CPU trial: how do (a) glyph-count-weighted font sampling with repeats and (b) finite queue size change participation
ratio relative to the uniform per-font corpus PR?  Font vectors = embeddings/all.json (39k fonts); per-font glyph counts g_f
from the bitmap file names (48px folders).  Training draws a PAIR uniformly, so font f is drawn with probability ~ g_f^2.

    OMP_NUM_THREADS=2 python experiments/embedding-geometry/queue_vs_corpus_pr.py
"""
import os, sys, json, collections
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
torch.set_num_threads(2)

counts = collections.Counter()
for d in ("google/bitmaps", "dafont/bitmaps", "dataset/smallimage"):
    for fn in os.listdir(d):
        counts[fn.rsplit(" ", 1)[0]] += 1          # "<font> <style> <glyph><l|u>.bmp" -> font key
emb = json.load(open("embeddings/all.json"))
keys = [k for k in emb if k in counts]
print(f"fonts in all.json: {len(emb)}; with a glyph count: {len(keys)}")
V = np.array([emb[k] for k in keys], dtype=np.float64)
g = np.array([counts[k] for k in keys], dtype=np.float64)
print(f"glyphs per font: mean {g.mean():.1f} median {np.median(g):.0f} p90 {np.percentile(g,90):.0f} max {g.max():.0f}")

def pr_w(X, w=None):
    w = np.ones(len(X)) / len(X) if w is None else w / w.sum()
    mu = w @ X; Xc = X - mu
    C = (Xc * w[:, None]).T @ Xc
    return np.trace(C) ** 2 / (C ** 2).sum()

N = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
M = 16384
for label, X in (("all.json vectors as stored", V), ("L2-normalised (what queue keys are)", N(V))):
    p = g ** 2 / (g ** 2).sum()
    kish = 1 / (p ** 2).sum()
    print(f"\n{label}")
    print(f"  uniform corpus PR                        : {pr_w(X):6.2f}")
    print(f"  exact population PR, weights ~ g^2       : {pr_w(X, g**2):6.2f}   (font-repeat weighting, no sampling noise)")
    print(f"  exact population PR, weights ~ g         : {pr_w(X, g):6.2f}")
    rng = np.random.RandomState(0)
    draws, distinct, prs = 20, [], []
    for _ in range(draws):
        idx = rng.choice(len(X), M, replace=True, p=p)
        distinct.append(len(np.unique(idx)))
        Xs = torch.tensor(X[idx]); Xc = Xs - Xs.mean(0)
        C = (Xc.T @ Xc / (M - 1)).numpy()
        prs.append(np.trace(C) ** 2 / (C ** 2).sum())
    print(f"  simulated queue (M={M}, ~g^2, 20 draws)   : {np.mean(prs):6.2f} +- {np.std(prs):.2f}")
    print(f"     distinct fonts in a queue: {np.mean(distinct):.0f} of {len(X)}   Kish n_eff of weights: {kish:.0f}")
    idx_u = [rng.choice(len(X), M, replace=True) for _ in range(20)]
    pu = []
    for idx in idx_u:
        Xs = torch.tensor(X[idx]); Xc = Xs - Xs.mean(0); C = (Xc.T @ Xc / (M - 1)).numpy(); pu.append(np.trace(C) ** 2 / (C ** 2).sum())
    print(f"  simulated queue, UNIFORM font draws (M={M}): {np.mean(pu):6.2f} +- {np.std(pu):.2f}   (finite-sample/duplicates only)")
