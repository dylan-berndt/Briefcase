"""
Where does the variance live? For a checkpoint, embed N random fonts x 26 lowercase letters (CPU) and report
participation ratio of (a) per-glyph L2-normalized embeddings, (b) per-font mean-pooled vectors (what all.json is),
(c) fraction of total per-glyph variance that is between-font vs. within-font (i.e. letter identity / noise).

    python experiments/embedding-geometry/glyph_vs_font_variance.py <checkpointDir> <suffix or ""> [nFonts]
"""
import os, sys, pickle
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from utils.vit import ViT
from utils.loaders.standard import loadImage

ckpt, suffix = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
torch.set_num_threads(4)
model, config = ViT.load(ckpt); model.eval()
pm = pickle.load(open("embeddings/fontGlyphPaths.pkl", "rb"))
letters = "abcdefghijklmnopqrstuvwxyz"
rng = np.random.RandomState(0)
names = [k for k in pm if all(l in pm[k] for l in letters)]
names = list(rng.choice(names, n, replace=False))

def fix(p):
    h, t = os.path.split(p); s, f = os.path.split(h)
    return os.path.join(s, f + suffix, t)

E = []
kept = 0
with torch.no_grad():
    for name in names:
        paths = [fix(pm[name][l]) for l in letters]
        if not all(os.path.exists(p) for p in paths): continue
        imgs = torch.stack([torch.tensor(loadImage(p)[1], dtype=torch.float32) for p in paths]).unsqueeze(-1)
        _, e = model(imgs)
        E.append(torch.nn.functional.normalize(e, dim=-1).numpy()); kept += 1
E = np.stack(E)  # [fonts, 26, D]
def pr(x):
    x = x - x.mean(0, keepdims=True)
    ev = np.linalg.svd(x, compute_uv=False) ** 2 / (len(x) - 1)
    return ev.sum() ** 2 / (ev ** 2).sum()
flat = E.reshape(-1, E.shape[-1])
fontMean = E.mean(1)
between = E.mean(1, keepdims=True); within = E - between
totalVar = ((flat - flat.mean(0)) ** 2).sum()
print(f"{ckpt} suffix={suffix!r}: {kept} fonts")
print(f"  PR per-glyph normalized      : {pr(flat):6.1f}")
print(f"  PR font-mean (all.json-style): {pr(fontMean):6.1f}")
print(f"  PR of within-font residuals  : {pr(within.reshape(-1, E.shape[-1])):6.1f}")
print(f"  variance share between-font  : {((between - flat.mean(0)) ** 2).sum() * E.shape[1] / totalVar:.3f}")
print(f"  mean |font-mean| norm        : {np.linalg.norm(fontMean, axis=1).mean():.3f}")
