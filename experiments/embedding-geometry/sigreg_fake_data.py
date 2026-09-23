"""
Controlled tests of what SIGReg does and doesn't see, on synthetic embeddings.

  A. Loss value vs. true rank, at the training batch size (64) and larger. Data live on a sphere of radius
     sqrt(512) inside a k-dim subspace (the ViT output is a LayerNorm output, so |x| ~ sqrt(512)); k=512 is isotropic.
  B. Optimisation: free points, started low-rank, minimised by SIGReg alone on random batches; effective rank tracked.
  C. Real checkpoint: SIGReg value and gradient magnitudes (SIGReg vs InfoNCE) on real 64-glyph batches.

    python experiments/embedding-geometry/sigreg_fake_data.py [A|B|C ...]
"""
import os, sys, pickle
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from utils.training import sigreg_strong_loss, sigreg_weak_loss, EmbeddingLoss

torch.manual_seed(0)
D = 512


def pr(x):
    x = x - x.mean(0, keepdim=True)
    c = x.T @ x / (len(x) - 1)
    return (torch.trace(c) ** 2 / (c ** 2).sum()).item()


def lowRankSphere(n, k):
    Q, _ = torch.linalg.qr(torch.randn(D, k))
    z = torch.nn.functional.normalize(torch.randn(n, k), dim=-1) * D ** 0.5
    return z @ Q.T


def expA():
    print("A. SIGReg value vs true rank (mean +- std over 30 fresh batches)")
    print(f"{'data':22s}" + "".join(f"  N={n:<14d}" for n in (64, 256, 1024)) + "  (strong, sketch 1024)")
    for label, gen in [("gaussian N(0,I) 512d", lambda n: torch.randn(n, D))] + \
                      [(f"sphere rank {k}", (lambda k: lambda n: lowRankSphere(n, k))(k)) for k in (4, 8, 16, 32, 64, 128, 512)]:
        row = f"{label:22s}"
        for n in (64, 256, 1024):
            v = torch.tensor([sigreg_strong_loss(gen(n), 1024).item() for _ in range(30)])
            row += f"  {v.mean():6.2f}+-{v.std():4.2f}   "
        print(row)
    print("   weak (frobenius) at N=64:  " + "  ".join(
        f"{lab}={torch.tensor([sigreg_weak_loss(g(64)).item() for _ in range(10)]).mean():.1f}"
        for lab, g in [("gauss", lambda n: torch.randn(n, D)), ("rank8", lambda n: lowRankSphere(n, 8)), ("rank64", lambda n: lowRankSphere(n, 64))]))


def expB():
    print("B. Minimising SIGReg alone from a rank-8 start (2048 free points, random batches); effective rank of ALL points")
    for name, fn in [("strong", lambda x: sigreg_strong_loss(x, 1024)), ("weak", sigreg_weak_loss)]:
        for batch in (64, 256):
            X = lowRankSphere(2048, 8).clone().requires_grad_(True)
            opt = torch.optim.Adam([X], lr=0.02)
            trace = []
            for step in range(1501):
                if step % 300 == 0:
                    trace.append(f"{step}:{pr(X.detach()):.1f}")
                idx = torch.randperm(2048)[:batch]
                loss = fn(X[idx]); opt.zero_grad(); loss.backward(); opt.step()
            print(f"  {name:6s} batch {batch:3d}  PR by step  " + "  ".join(trace))


def expC():
    from utils.vit import ViT
    from utils.loaders.standard import loadImage
    ckpt = os.path.join("checkpoints", "pretrain", "strong-sigreg-64px-patch8")
    model, _ = ViT.load(ckpt); model.eval()
    pm = pickle.load(open("embeddings/fontGlyphPaths.pkl", "rb"))
    letters = list("abcdefghijklmnopqrstuvwxyz")
    names = [k for k in pm if all(l in pm[k] for l in letters)]
    rng = np.random.RandomState(1)
    fix = lambda p: os.path.join(*os.path.split(os.path.split(p)[0])[:1], os.path.split(os.path.split(p)[0])[1] + "_64", os.path.split(p)[1])
    loss = EmbeddingLoss()
    rows = []
    for _ in range(8):
        fonts = rng.choice(names, 64, replace=False)
        ims = [[torch.tensor(loadImage(fix(pm[f][l]))[1], dtype=torch.float32) for l in rng.choice(letters, 2, replace=False)] for f in fonts]
        a = torch.stack([i[0] for i in ims]).unsqueeze(-1); b = torch.stack([i[1] for i in ims]).unsqueeze(-1)
        with torch.no_grad():
            pass
        _, x = model(a); _, y = model(b)
        x = x.detach().requires_grad_(True); y = y.detach().requires_grad_(True)
        fam = torch.arange(64)
        out = loss(x, y, fam)  # in-batch only
        gInfo = torch.autograd.grad(out["info"], [x, y], retain_graph=True)
        gSig = torch.autograd.grad(0.1 * out["sig"], [x, y])
        gi = sum(g.norm() ** 2 for g in gInfo).sqrt().item(); gs = sum(g.norm() ** 2 for g in gSig).sqrt().item()
        rows.append((out["sig"].item(), out["info"].item(), gi, gs, x.detach().norm(dim=1).mean().item(), pr(x.detach()), pr(torch.cat([x, y]).detach())))
    r = np.array(rows).mean(0)
    print("C. real step-latest checkpoint, 8 batches of 64 fonts x 2 glyphs (in-batch InfoNCE, no queue)")
    print(f"   SIGReg={r[0]:.2f}  InfoNCE={r[1]:.2f}  |grad InfoNCE|={r[2]:.4f}  |grad 0.1*SIGReg|={r[3]:.4f}  ratio sig/info={r[3]/r[2]:.2f}")
    print(f"   mean |x|={r[4]:.1f} (sqrt512=22.6)  PR of batch(64)={r[5]:.1f}  PR of batch(128)={r[6]:.1f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C"]
    for w in which:
        {"A": expA, "B": expB, "C": expC}[w]()
