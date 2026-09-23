"""
On real 64-font batches (2 glyphs per font) from a checkpoint: does each loss term's descent direction raise or lower
the effective rank of the batch's embeddings? Reports cos(-grad term, grad PR) w.r.t. the raw embeddings, plus each
term's gradient norm. InfoNCE here is in-batch only (no MoCo queue).

    python experiments/embedding-geometry/gradient_vs_rank.py [checkpointDir]
"""
import os, sys, pickle
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from utils.vit import ViT
from utils.loaders.standard import loadImage
from utils.training import EmbeddingLoss

ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join("checkpoints", "pretrain", "strong-sigreg-64px-patch8")
model, _ = ViT.load(ckpt); model.eval()
pm = pickle.load(open("embeddings/fontGlyphPaths.pkl", "rb"))
letters = list("abcdefghijklmnopqrstuvwxyz")
names = [k for k in pm if all(l in pm[k] for l in letters)]
rng = np.random.RandomState(2)
suffix = os.environ.get("CACHE_SUFFIX", "_64")
def fix(p):
    h, t = os.path.split(p); s, f = os.path.split(h); return os.path.join(s, f + suffix, t)

def prOf(z):
    z = z - z.mean(0, keepdim=True); c = z.T @ z / (len(z) - 1)
    return torch.trace(c) ** 2 / (c ** 2).sum()

loss = EmbeddingLoss()
res = []
for _ in range(8):
    fonts = rng.choice(names, 64, replace=False)
    pairs = [rng.choice(letters, 2, replace=False) for _ in fonts]
    a = torch.stack([torch.tensor(loadImage(fix(pm[f][l[0]]))[1], dtype=torch.float32) for f, l in zip(fonts, pairs)]).unsqueeze(-1)
    b = torch.stack([torch.tensor(loadImage(fix(pm[f][l[1]]))[1], dtype=torch.float32) for f, l in zip(fonts, pairs)]).unsqueeze(-1)
    with torch.no_grad():
        _, x = model(a); _, y = model(b)
    x = x.clone().requires_grad_(True); y = y.clone().requires_grad_(True)
    out = loss(x, y, torch.arange(64))
    gInfo = torch.cat([g.flatten() for g in torch.autograd.grad(out["info"], [x, y], retain_graph=True)])
    gSig = torch.cat([g.flatten() for g in torch.autograd.grad(0.1 * out["sig"], [x, y], retain_graph=True)])
    p = prOf(torch.cat([x, y]))
    gPr = torch.cat([g.flatten() for g in torch.autograd.grad(p, [x, y])])
    cos = lambda u, v: torch.nn.functional.cosine_similarity(u, v, dim=0).item()
    res.append((cos(-gInfo, gPr), cos(-gSig, gPr), gInfo.norm().item(), gSig.norm().item(), p.item()))
r = np.array(res)
print(f"{ckpt}")
print(f"  cos(-grad InfoNCE, grad PR)      = {r[:,0].mean():+.3f}  (+ raises rank, - lowers)")
print(f"  cos(-grad 0.1*SIGReg, grad PR)   = {r[:,1].mean():+.3f}")
print(f"  |grad InfoNCE|={r[:,2].mean():.4f}  |grad 0.1*SIGReg|={r[:,3].mean():.4f}   batch PR={r[:,4].mean():.1f}")
