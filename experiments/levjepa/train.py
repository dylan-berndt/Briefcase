"""
LeVJEPA-style pretraining on font glyphs (see "Planned investigation: LeVJEPA-style font pretraining" in CLAUDE.md).

One encoder + a projector, an invariance loss between a global view and local views, SIGReg against collapse. No
predictor, target encoder, stop-gradient, momentum encoder, queue or negatives.

  frame = glyph. A font is 52 frames (a-z = 0-25, A-Z = 26-51) of 6x6 patches of 8x8 pixels (48px glyph images from
  dataset/levjepa, built by build_glyph_cache.py). Every token carries three rotary coordinates (letter, row, col);
  (row, col) are the patch's position inside its glyph and are never shifted, so a token looks positionally the same
  in every view. There are no absolute position embeddings.
  global view = all of the font's glyphs; keep --keep of all patch tokens, sampled jointly across glyphs (--inkFrac of
                the kept tokens from patches that contain ink, the rest uniform over all present patches).
  local views = --views spatial crops of --cropPatches x --cropPatches patches (patch-aligned, native resolution), one
                random window per view shared by every glyph of the font, same --keep. Positions unchanged.
  CLS token: attends to every token (and itself), is not attended to. Patch tokens attend to patch tokens only.
  loss = L_inv + lambda * SIGReg, L_inv = 1/(V+1) sum_v ||z_0 - z_v||^2 with gradients through both sides, SIGReg
  (utils.training.sigreg_strong_loss, Epps-Pulley, 1024 directions, 17 knots) applied to every view's projector output.

Evaluation-time embedding (no token dropping, 52-glyph global view) is deferred until after pretraining.

    python -u experiments/levjepa/train.py --supervise          # the real run; restarts itself and resumes
    python -u experiments/levjepa/train.py --maxSteps 20 --noWandb --runName smoke --batch 32   # quick check

Writes ONLY to checkpoints/pretrain/<runName>; never touches pretrain/latest or pretrain/best. checkpoint.pt is the bare
encoder state dict (CLS output = post-final-LayerNorm backbone feature, the thing that gets embedded downstream).
"""
import argparse
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time

_scriptDir = os.path.dirname(os.path.abspath(__file__))
_repoRoot = os.path.dirname(os.path.dirname(_scriptDir))
os.chdir(_repoRoot)  # data/checkpoint paths are repo-root-relative
if _repoRoot not in sys.path:
    sys.path.insert(0, _repoRoot)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.training import sigreg_strong_loss

NUM_GLYPHS = 52
GRID = 6            # patches per glyph side (48px / 8px)
PATCH = 8
PATCH_PIXELS = PATCH * PATCH
PER_GLYPH = GRID * GRID
ROPE_BASES = (200.0, 20.0, 20.0)   # letter, row, col: geometric frequencies from 1 down to 1/base


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--runName", default="levjepa-d256")
    p.add_argument("--data", default=os.path.join("dataset", "levjepa"))
    # model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlpRatio", type=int, default=3)
    p.add_argument("--projHidden", type=int, default=2048)
    p.add_argument("--projDim", type=int, default=256)
    # views
    p.add_argument("--keep", type=float, default=0.10, help="fraction of patch tokens kept per view")
    p.add_argument("--inkFrac", type=float, default=0.75, help="fraction of the kept tokens drawn from ink patches")
    p.add_argument("--inkThreshold", type=float, default=0.1)
    p.add_argument("--views", type=int, default=4, help="local views per font")
    p.add_argument("--cropPatches", type=int, default=3, help="local crop side in patches (3 = 24px)")
    p.add_argument("--minGlyphs", type=int, default=26, help="fonts with fewer present glyphs are not used")
    # objective
    p.add_argument("--lam", type=float, default=0.02, help="SIGReg weight (LeVJEPA)")
    p.add_argument("--invReduction", choices=["mean", "sum"], default="mean",
                   help="reduce ||z0 - zv||^2 over embedding dims by mean (LeJEPA reference code) or sum (paper eq. 2)")
    p.add_argument("--sketch", type=int, default=1024, help="SIGReg random directions")
    # optimisation
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--noCkpt", action="store_true", help="disable per-block gradient checkpointing")
    p.add_argument("--vramFraction", type=float, default=0.93)
    # data split / run control
    p.add_argument("--holdout", type=float, default=0.05)
    p.add_argument("--splitSalt", default="levjepa-split-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--saveEvery", type=int, default=500)
    p.add_argument("--evalEvery", type=int, default=500)
    p.add_argument("--evalBatches", type=int, default=4)
    p.add_argument("--diagEvery", type=int, default=50)
    p.add_argument("--maxSteps", type=int, default=None, help="stop (and save) after this many total steps")
    p.add_argument("--limitFonts", type=int, default=None, help="use only the first N usable fonts (smoke tests)")
    p.add_argument("--noWandb", action="store_true")
    p.add_argument("--wandbEntity", default="dylanberndt123-missouri-state-university")
    p.add_argument("--wandbProject", default="Briefcase")
    p.add_argument("--supervise", action="store_true", help="re-launch this script until it exits cleanly")
    return p.parse_args()


# ------------------------------------------------------------------ data

class GlyphStore:
    """dataset/levjepa: glyphs.npy uint8 [fonts, 52, 48, 48] (memmap), present.npy bool [fonts, 52], meta.json."""

    def __init__(self, dataDir, holdout, salt, minGlyphs, limit=None):
        self.glyphs = np.load(os.path.join(dataDir, "glyphs.npy"), mmap_mode="r")
        self.present = np.load(os.path.join(dataDir, "present.npy"))
        with open(os.path.join(dataDir, "meta.json"), encoding="utf8") as f:
            self.meta = json.load(f)
        assert self.glyphs.shape[1:] == (NUM_GLYPHS, GRID * PATCH, GRID * PATCH), self.glyphs.shape
        names = self.meta["names"]
        usable = np.where(self.present.sum(1) >= minGlyphs)[0]
        if limit:
            usable = usable[:limit]

        def trainBucket(name):
            digest = hashlib.sha256(f"{salt}:{name}".encode("utf-8")).hexdigest()
            return int(digest[:8], 16) / 0xFFFFFFFF >= holdout

        isTrain = np.array([trainBucket(names[i]) for i in usable])
        self.train, self.test = usable[isTrain], usable[~isTrain]
        print(f"fonts: {len(names)} in cache, {len(usable)} with >= {minGlyphs} glyphs -> "
              f"{len(self.train)} train / {len(self.test)} held out", flush=True)

    def fetch(self, idx):
        idx = np.sort(idx)  # ascending reads are friendlier to the memmap
        return torch.from_numpy(np.ascontiguousarray(self.glyphs[idx])), torch.from_numpy(self.present[idx])


def trainBatches(store, batch, seed, startStep, stepsPerEpoch):
    """Endless (glyphs uint8, present bool) batches: a fresh permutation of the train fonts per epoch (seeded by epoch,
    so a resumed run continues the same order), dropping the last partial batch."""
    step, cachedEpoch, perm = startStep, None, None
    while True:
        epoch, offset = divmod(step, stepsPerEpoch)
        if epoch != cachedEpoch:
            cachedEpoch, perm = epoch, np.random.RandomState(seed * 100003 + epoch).permutation(store.train)
        yield store.fetch(perm[offset * batch:(offset + 1) * batch])
        step += 1


class ThreadPrefetcher:
    """Runs the (memmap-reading) batch producer in a background thread so loading overlaps GPU compute."""

    def __init__(self, iterable, depth=3):
        self.iterable, self.depth = iterable, depth

    def __iter__(self):
        buffer = queue.Queue(maxsize=self.depth)
        done = object()

        def produce():
            try:
                for item in self.iterable:
                    buffer.put(item)
            except BaseException as error:
                buffer.put(error)
            finally:
                buffer.put(done)

        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = buffer.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


# ------------------------------------------------------------------ views (all on the GPU)

class ViewSampler:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        pos = torch.arange(PER_GLYPH, device=device)
        self.row, self.col = pos // GRID, pos % GRID
        self.kGlobal = int(round(cfg.keep * NUM_GLYPHS * PER_GLYPH))
        self.kLocal = int(round(cfg.keep * NUM_GLYPHS * cfg.cropPatches ** 2))

    def patchify(self, glyphs):
        """glyphs uint8 [B, 52, 48, 48] -> patches float [B, 52*36, 64], ink flags bool [B, 52*36]."""
        x = glyphs.float() / 255.0
        B = x.shape[0]
        p = x.view(B, NUM_GLYPHS, GRID, PATCH, GRID, PATCH).permute(0, 1, 2, 4, 3, 5)
        p = p.reshape(B, NUM_GLYPHS * PER_GLYPH, PATCH_PIXELS)
        return p, p.amax(-1) > self.cfg.inkThreshold

    def select(self, cand, ink, K):
        """K token indices per font out of the candidate mask: inkFrac of them from ink patches (falling back to other
        candidates if a font has too few), the rest uniform over the remaining candidates. Never picks a non-candidate."""
        B, N = cand.shape
        kInk = int(round(K * self.cfg.inkFrac))
        neg = torch.full((B, N), -10.0, device=cand.device)
        r = torch.rand(B, N, device=cand.device)
        s1 = torch.where(cand & ink, r, torch.where(cand, r * 0.5 - 1.0, neg))
        idx1 = s1.topk(kInk, dim=1).indices
        s2 = torch.where(cand, torch.rand(B, N, device=cand.device), neg)
        s2.scatter_(1, idx1, -10.0)
        idx2 = s2.topk(K - kInk, dim=1).indices
        return torch.cat([idx1, idx2], dim=1)

    def gather(self, patches, idx):
        tokens = patches.gather(1, idx.unsqueeze(-1).expand(-1, -1, PATCH_PIXELS))
        pos = idx % PER_GLYPH
        coords = torch.stack([(idx // PER_GLYPH), pos // GRID, pos % GRID], dim=-1).float()
        return tokens, coords

    def globalView(self, patches, ink, present):
        B = patches.shape[0]
        cand = present.unsqueeze(-1).expand(-1, -1, PER_GLYPH).reshape(B, -1)
        return self.gather(patches, self.select(cand, ink, self.kGlobal))

    def localView(self, patches, ink, present):
        B, c = patches.shape[0], self.cfg.cropPatches
        r0 = torch.randint(0, GRID - c + 1, (B, 1), device=self.device)
        c0 = torch.randint(0, GRID - c + 1, (B, 1), device=self.device)
        inWindow = (self.row >= r0) & (self.row < r0 + c) & (self.col >= c0) & (self.col < c0 + c)   # [B, 36]
        cand = (present.unsqueeze(-1) & inWindow.unsqueeze(1)).reshape(B, -1)
        return self.gather(patches, self.select(cand, ink, self.kLocal))

    @torch.no_grad()
    def sample(self, glyphs, present):
        patches, ink = self.patchify(glyphs)
        views = [self.globalView(patches, ink, present)]
        views += [self.localView(patches, ink, present) for _ in range(self.cfg.views)]
        return views


# ------------------------------------------------------------------ model

def ropeFrequencies(headDim):
    """Split the head dimension over (letter, row, col): about 1/4 for the letter axis, the rest evenly."""
    nLetter = (headDim // 8) * 2
    nSpatial = (headDim - nLetter) // 2
    dims = (nLetter, nSpatial, headDim - nLetter - nSpatial)
    assert all(d > 0 and d % 2 == 0 for d in dims), f"head dim {headDim} cannot be split into even rotary blocks {dims}"
    return [ROPE_BASES[a] ** (-torch.arange(d // 2).float() / (d // 2)) for a, d in enumerate(dims)]


def applyRope(x, cos, sin):
    """x [B, H, K, hd]; cos/sin [B, 1, K, hd/2]; rotates consecutive (even, odd) pairs."""
    x = x.unflatten(-1, (-1, 2))
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlpRatio):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlpRatio), nn.GELU(), nn.Linear(dim * mlpRatio, dim))

    def forward(self, h, cos, sin):
        """h [B, 1+K, D]: slot 0 is CLS. Patch tokens attend to patch tokens only; CLS attends to everything."""
        B, L, D = h.shape
        q, k, v = self.qkv(self.n1(h)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        qp, kp, vp = applyRope(q[:, :, 1:], cos, sin), applyRope(k[:, :, 1:], cos, sin), v[:, :, 1:]
        patchOut = F.scaled_dot_product_attention(qp, kp, vp)
        clsOut = F.scaled_dot_product_attention(q[:, :, :1], torch.cat([k[:, :, :1], kp], dim=2),
                                                torch.cat([v[:, :, :1], vp], dim=2))
        a = torch.cat([clsOut, patchOut], dim=2).transpose(1, 2).reshape(B, L, D)
        h = h + self.out(a)
        return h + self.mlp(self.n2(h))


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlpRatio):
        super().__init__()
        self.embed = nn.Linear(PATCH_PIXELS, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlpRatio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        for a, f in enumerate(ropeFrequencies(dim // heads)):
            self.register_buffer(f"freq{a}", f, persistent=False)
        self.useCkpt = True

    def rope(self, coords):
        ang = torch.cat([coords[..., a:a + 1] * getattr(self, f"freq{a}") for a in range(3)], dim=-1)
        return ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)

    def forward(self, tokens, coords):
        """tokens [B, K, 64] pixels, coords [B, K, 3] -> CLS feature [B, D] (after the final LayerNorm)."""
        h = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), self.embed(tokens)], dim=1)
        cos, sin = self.rope(coords)
        for block in self.blocks:
            if self.useCkpt and self.training:
                h = checkpoint(block, h, cos, sin, use_reentrant=False)
            else:
                h = block(h, cos, sin)
        return self.norm(h[:, 0])


class Projector(nn.Sequential):
    def __init__(self, dim, hidden, out):
        super().__init__(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, out))


# ------------------------------------------------------------------ objective and diagnostics

def levjepaLoss(z, cfg):
    """z: list of projector outputs [B, K], index 0 = global view. Returns total, invariance, SIGReg (view mean)."""
    z0, zs = z[0], torch.stack(z[1:])
    dist = (z0.unsqueeze(0) - zs).pow(2)
    dist = dist.mean(-1) if cfg.invReduction == "mean" else dist.sum(-1)      # [V, B]
    inv = dist.sum(0).mean() / (len(z))                                        # 1/(V+1) sum_v, v = 0 contributes 0
    sig = torch.stack([sigreg_strong_loss(v, sketch_dim=cfg.sketch) for v in z]).mean()
    return inv + cfg.lam * sig, inv, sig


@torch.no_grad()
def participationRatio(x):
    x = x.detach().float()
    lam = torch.linalg.svdvals(x - x.mean(0, keepdim=True)).square()
    return (lam.sum().square() / lam.square().sum()).item()


@torch.no_grad()
def retrieval(local, glob):
    """In-batch: does a local view's embedding find its own font's global-view embedding? -> (R@1, R@5)."""
    sim = F.normalize(local.float(), dim=-1) @ F.normalize(glob.float(), dim=-1).T
    rank = (sim > sim.diag().unsqueeze(1)).sum(1)
    return (rank == 0).float().mean().item(), (rank < 5).float().mean().item()


@torch.no_grad()
def diagnostics(feats, z, prefix):
    out = {f"{prefix}/PR backbone": participationRatio(feats[0]), f"{prefix}/PR projector": participationRatio(z[0]),
           f"{prefix}/embedStd projector": z[0].float().std(0).mean().item()}
    r1 = [retrieval(f, feats[0]) for f in feats[1:]]
    p1 = [retrieval(v, z[0]) for v in z[1:]]
    out[f"{prefix}/local->global R@1 backbone"] = float(np.mean([a for a, _ in r1]))
    out[f"{prefix}/local->global R@5 backbone"] = float(np.mean([b for _, b in r1]))
    out[f"{prefix}/local->global R@1 projector"] = float(np.mean([a for a, _ in p1]))
    return out


def forwardViews(encoder, projector, views):
    feats = [encoder(t, c) for t, c in views]
    return feats, [projector(f) for f in feats]


# ------------------------------------------------------------------ run state

def atomicSave(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def saveState(dirPath, cfg, encoder, projector, optimizer, step, runId):
    original = signal.signal(signal.SIGINT, signal.SIG_IGN)   # never die half-way through a save
    try:
        os.makedirs(dirPath, exist_ok=True)
        atomicSave(encoder.state_dict(), os.path.join(dirPath, "checkpoint.pt"))
        with open(os.path.join(dirPath, "config.json"), "w") as f:
            json.dump({"model": {"type": "levjepa", "dim": cfg.dim, "depth": cfg.depth, "heads": cfg.heads,
                                 "mlpRatio": cfg.mlpRatio, "patchSize": PATCH, "imageSize": GRID * PATCH,
                                 "ropeBases": ROPE_BASES, "glyphs": NUM_GLYPHS},
                       "train": vars(cfg)}, f, indent=1)
        atomicSave({"step": step, "optimizer": optimizer.state_dict(), "projector": projector.state_dict(),
                    "wandbId": runId}, os.path.join(dirPath, "train_state.pt"))
        with open(os.path.join(dirPath, "progress.json.tmp"), "w") as f:
            json.dump({"step": step, "time": time.time()}, f)
        os.replace(os.path.join(dirPath, "progress.json.tmp"), os.path.join(dirPath, "progress.json"))
    finally:
        signal.signal(signal.SIGINT, original)


def supervise(cfg):
    progress = os.path.join("checkpoints", "pretrain", cfg.runName, "progress.json")

    def savedStep():
        try:
            with open(progress) as f:
                return json.load(f)["step"]
        except Exception:
            return -1

    args = [a for a in sys.argv[1:] if a != "--supervise"]
    failures = attempt = 0
    while failures < 30:
        attempt += 1
        before = savedStep()
        print(f"[supervisor] attempt {attempt}, saved step {before}, {time.strftime('%H:%M:%S')}", flush=True)
        code = subprocess.call([sys.executable, "-u", os.path.abspath(__file__)] + args)
        after = savedStep()
        print(f"\n[supervisor] exit code {code}, saved step {after}", flush=True)
        if code == 0:
            break
        failures = 0 if after > before else failures + 1
        time.sleep(min(60, 10 * (failures + 1)))
    print("[supervisor] finished", flush=True)


def lrAt(step, total, cfg):
    warm = max(1, int(cfg.warmup * total))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def evaluate(encoder, projector, sampler, store, cfg, device):
    if len(store.test) < cfg.batch:
        return {}
    encoder.eval(); projector.eval()
    sums, n = {}, 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        torch.manual_seed(1234)
        rng = np.random.RandomState(1234)
        for _ in range(cfg.evalBatches):
            glyphs, present = store.fetch(rng.choice(store.test, cfg.batch, replace=False))
            views = sampler.sample(glyphs.to(device), present.to(device))
            feats, z = forwardViews(encoder, projector, views)
            total, inv, sig = levjepaLoss(z, cfg)
            row = {"LeVJEPA/test/loss": total.item(), "LeVJEPA/test/inv": inv.item(), "LeVJEPA/test/sigreg": sig.item()}
            row.update(diagnostics(feats, z, "LeVJEPA/test"))
            for k, v in row.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
    encoder.train(); projector.train()
    return {k: v / n for k, v in sums.items()}


def main():
    cfg = parseArgs()
    if cfg.supervise:
        return supervise(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.vramFraction)   # overflow raises OOM instead of spilling to RAM
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    store = GlyphStore(cfg.data, cfg.holdout, cfg.splitSalt, cfg.minGlyphs, cfg.limitFonts)
    stepsPerEpoch = len(store.train) // cfg.batch
    total = stepsPerEpoch * cfg.epochs
    sampler = ViewSampler(cfg, device)
    print(f"tokens per view: global {sampler.kGlobal}, local {sampler.kLocal} x {cfg.views}; "
          f"{stepsPerEpoch} steps/epoch, {total} steps total", flush=True)

    encoder = Encoder(cfg.dim, cfg.depth, cfg.heads, cfg.mlpRatio).to(device)
    encoder.useCkpt = not cfg.noCkpt
    projector = Projector(cfg.dim, cfg.projHidden, cfg.projDim).to(device)
    print(f"encoder {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M params, "
          f"projector {sum(p.numel() for p in projector.parameters()) / 1e6:.2f}M", flush=True)

    decay, noDecay = [], []
    for module in (encoder, projector):
        for name, prm in module.named_parameters():
            (decay if prm.ndim >= 2 and "cls" not in name else noDecay).append(prm)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd}, {"params": noDecay, "weight_decay": 0.0}],
                                  lr=cfg.lr)

    runDir = os.path.join("checkpoints", "pretrain", cfg.runName)
    statePath = os.path.join(runDir, "train_state.pt")
    step, runId = 0, None
    if os.path.exists(statePath):
        state = torch.load(statePath, map_location=device, weights_only=False)
        encoder.load_state_dict(torch.load(os.path.join(runDir, "checkpoint.pt"), map_location=device, weights_only=True))
        projector.load_state_dict(state["projector"])
        optimizer.load_state_dict(state["optimizer"])
        step, runId = state["step"], state["wandbId"]
        print(f"RESUMING from step {step} (wandb run {runId})", flush=True)

    run = None
    if not cfg.noWandb:
        import wandb
        run = wandb.init(entity=cfg.wandbEntity, project=cfg.wandbProject, name=cfg.runName, config=vars(cfg),
                         id=runId, resume="allow")
        runId = run.id

    encoder.train(); projector.train()
    stop = cfg.maxSteps if cfg.maxSteps is not None else total
    stop = min(stop, total)
    batches = iter(ThreadPrefetcher(trainBatches(store, cfg.batch, cfg.seed, step, stepsPerEpoch), depth=3))
    started, stepTimes = time.time(), []
    try:
        while step < stop:
            t0 = time.time()
            glyphs, present = next(batches)
            views = sampler.sample(glyphs.to(device, non_blocking=True), present.to(device, non_blocking=True))
            lr = lrAt(step, total, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            feats, z = forwardViews(encoder, projector, views)
            loss, inv, sig = levjepaLoss(z, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradNorm = torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projector.parameters()), cfg.clip)
            optimizer.step()

            logged = {"LeVJEPA/loss": loss.item(), "LeVJEPA/inv": inv.item(), "LeVJEPA/sigreg": sig.item(),
                      "LeVJEPA/lr": lr, "LeVJEPA/gradNorm": gradNorm.item(), "LeVJEPA/epoch": step / stepsPerEpoch}
            heavy = step % cfg.diagEvery == 0 or step % cfg.evalEvery == 0
            if step % cfg.diagEvery == 0:
                logged.update(diagnostics([f.detach() for f in feats], [v.detach() for v in z], "LeVJEPA/train"))
            if step % cfg.evalEvery == 0:
                logged.update(evaluate(encoder, projector, sampler, store, cfg, device))
            if heavy and run is None:   # no wandb: still show the diagnostics
                print("\n" + json.dumps({k.replace("LeVJEPA/", ""): round(v, 4) for k, v in logged.items()
                                        if "train/" in k or "test/" in k}), flush=True)
            step += 1
            stepTimes.append(time.time() - t0)
            logged["LeVJEPA/stepTime"] = stepTimes[-1]
            if run is not None:
                run.log(logged, step=step)
            if step % 20 == 0:
                recent = float(np.mean(stepTimes[-50:]))
                eta = (total - step) * recent / 3600
                print(f"\rstep {step}/{total} | loss {logged['LeVJEPA/loss']:.4f} inv {logged['LeVJEPA/inv']:.4f} "
                      f"sig {logged['LeVJEPA/sigreg']:.3f} | {recent:.2f}s/step | eta {eta:.1f}h", end="", flush=True)
            if step % cfg.saveEvery == 0:
                saveState(runDir, cfg, encoder, projector, optimizer, step, runId)
    except KeyboardInterrupt:
        pass
    saveState(runDir, cfg, encoder, projector, optimizer, step, runId)
    peak = f", peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB allocated" if device == "cuda" else ""
    print(f"\nsaved step {step} to {runDir} after {(time.time() - started) / 3600:.2f}h{peak}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
