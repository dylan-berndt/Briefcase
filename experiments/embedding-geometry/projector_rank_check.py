"""
Effective dimension (participation ratio) at each stage of a projector run, on real glyphs, plus a synthetic control.

  python experiments/embedding-geometry/projector_rank_check.py <runDir> <cacheSuffix>
Real: 1500 random fonts x 2 random lowercase glyphs -> backbone CLS c, raw projector output z, L2-normalised z, L2-normalised c.
Control: random-init projector (same layout) applied to fake features of KNOWN rank on the sphere, input PR vs output PR.
"""
import os, sys, pickle
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "experiments/embedding-geometry")
import numpy as np, torch
torch.set_num_threads(4)
import retrain_strong_sigreg as R
from utils.vit import ViT
from utils.loaders.standard import loadImage

def pr(x):
    x = x - x.mean(0, keepdim=True); c = x.T @ x / (len(x) - 1)
    return (torch.trace(c) ** 2 / (c * c).sum()).item()

runDir, suffix = sys.argv[1], sys.argv[2]
vit, cfg = ViT.load(runDir); vit.eval()
st = torch.load(os.path.join(runDir, "train_state.pt"), weights_only=False)
proj = R.ProjectedViT(vit, st["projDim"], R.PROJ_HIDDEN); proj.projector.load_state_dict(st["projector"]); proj.eval()
print(f"{runDir}: step {st['step']}, embedDim {cfg.model.embedDim}, projDim {st['projDim']}")
pm = pickle.load(open("embeddings/fontGlyphPaths.pkl", "rb")); L = list("abcdefghijklmnopqrstuvwxyz")
names = [k for k in pm if all(l in pm[k] for l in L)]; rng = np.random.RandomState(0)
fix = lambda p: os.path.join(os.path.split(os.path.split(p)[0])[0], os.path.split(os.path.split(p)[0])[1] + suffix, os.path.split(p)[1])
C, Z = [], []
with torch.no_grad():
    for f in rng.choice(names, 1500, replace=False):
        ims = torch.stack([torch.tensor(loadImage(fix(pm[f][l]))[1], dtype=torch.float32) for l in rng.choice(L, 2, replace=False)]).unsqueeze(-1)
        c, z = proj(ims); C.append(c); Z.append(z)
C, Z = torch.cat(C), torch.cat(Z); N = torch.nn.functional.normalize
print(f"real ({len(C)} glyphs, eval mode):")
print(f"  backbone c            PR {pr(C):6.1f}   (raw)   {pr(N(C,dim=-1)):6.1f} (L2-normalised)")
print(f"  projector output z    PR {pr(Z):6.1f}   (raw)   {pr(N(Z,dim=-1)):6.1f} (L2-normalised)   mean|z|={Z.norm(dim=1).mean():.1f}")
print("control: random-init projector on fake sphere features of known rank (d=256, projDim=256):")
torch.manual_seed(0)
for k in (4, 8, 12, 24, 64, 256):
    Q, _ = torch.linalg.qr(torch.randn(256, k)); x = N(torch.randn(3000, k), dim=-1) * 256 ** 0.5 @ Q.T
    p = torch.nn.Sequential(torch.nn.Linear(256, R.PROJ_HIDDEN), torch.nn.BatchNorm1d(R.PROJ_HIDDEN), torch.nn.GELU(), torch.nn.Linear(R.PROJ_HIDDEN, 256)).train()
    with torch.no_grad(): y = p(x)
    print(f"  input rank {k:3d}: PR in {pr(x):6.1f} -> out raw {pr(y):6.1f}, out normalised {pr(N(y,dim=-1)):6.1f}")

# EMA (momentum) model: what the logged queue diagnostics are actually computed from
import copy
ema = R.ProjectedViT(ViT(cfg.model), st["projDim"], R.PROJ_HIDDEN); ema.load_state_dict(st["momentum"]); ema.eval()
rng = np.random.RandomState(0); C2, Z2 = [], []
with torch.no_grad():
    for f in rng.choice(names, 1500, replace=False):
        ims = torch.stack([torch.tensor(loadImage(fix(pm[f][l]))[1], dtype=torch.float32) for l in rng.choice(L, 2, replace=False)]).unsqueeze(-1)
        c, z = ema(ims); C2.append(c); Z2.append(z)
C2, Z2 = torch.cat(C2), torch.cat(Z2)
print("EMA (momentum) model, same glyphs (BN running stats copied from the online model, weights are the EMA):")
print(f"  backbone c            PR {pr(C2):6.1f}   (raw)   {pr(N(C2,dim=-1)):6.1f} (L2-normalised)")
print(f"  projector output z    PR {pr(Z2):6.1f}   (raw)   {pr(N(Z2,dim=-1)):6.1f} (L2-normalised)   mean|z|={Z2.norm(dim=1).mean():.1f}")
