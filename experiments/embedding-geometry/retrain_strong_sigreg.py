"""
Re-runs pretraining (configs/vit.json: same font data, contrastive same-font task, MoCo queue
16384, Adam 1e-4) with the changes this investigation's evidence points at:

  * EmbeddingLoss defaults to sigreg_strong_loss (Epps-Pulley / empirical-characteristic-function
    test over 1024 random projections, as LeVJEPA uses) instead of sigreg_weak_loss, whose
    covariance-only target was shown to converge while nearest-neighbor cosine density stayed ~0.96.
  * imageSize 48 -> 64 (fontSize 43), patchSize 8: 36 -> 64 patches. Peak VRAM measured with a
    hard allocator cap (measure_vram.py): 4.25 GB allocated / 4.67 GB reserved at batch 64;
    batch 96/128 do not fit; imageSize 96 only fits at batch 32.
  * Bitmaps come from the size-suffixed "_64" cache folders built by build_bitmap_cache.py (the
    original 48px cache is untouched -- cached filenames do not encode resolution).

Operational hardening for unattended runs: atomic model + optimizer/momentum/step/wandb-id
checkpoints and automatic resume, a streaming sampler (DataLoader's shuffle=True materializes a
~116M-element Python list), background-thread batch prefetch, test metrics every TEST_EVERY steps,
a hard CUDA allocator cap (this driver silently spills VRAM into system RAM instead of raising
OOM), and startup assertions that the images really are 64px from the new cache.

Writes ONLY to checkpoints/pretrain/<RUN_NAME>; never touches pretrain/latest or pretrain/best.
Run it under run_supervised.py, which restarts it after any crash.

    python -u experiments/embedding-geometry/retrain_strong_sigreg.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)  # config/checkpoint/cache paths are repo-root-relative

import json
import queue
import signal
import threading
import time

import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from utils.config import Config
from utils.loaders.standard import loadImage
from utils.pretraining import PairedImageData, device
from utils.training import EmbeddingLoss, MoCoQueue, MomentumEncoder, recallAtK
from utils.vit import ViT

# Environment overrides let a short branch run start from another run's checkpoint without touching it:
#   RUN_NAME=<new dir/wandb name>  ALPHA=<sigreg weight>  INIT_FROM=<checkpoint dir with checkpoint.pt + train_state.pt>
RUN_NAME = _os.environ.get("RUN_NAME", "strong-sigreg-64px-patch8")
ALPHA = float(_os.environ.get("ALPHA", "0.1"))
INIT_FROM = _os.environ.get("INIT_FROM")
# Projector head (LeVJEPA arXiv 2608.27395 sec. 3 / LeJEPA 2511.08544): InfoNCE and SIGReg act on the projector output; the
# encoder's CLS feature (a LayerNorm output) is what gets embedded afterwards. PROJ_DIM=0 disables it (loss on the CLS directly).
SIGREG = _os.environ.get("SIGREG", "strong")   # "strong" (Epps-Pulley) or "weak" (covariance Frobenius), see utils/training.py
PROJ_DIM = int(_os.environ.get("PROJ_DIM", "256"))
EXCLUDE_FONTS_FILE = _os.environ.get("EXCLUDE_FONTS_FILE")   # optional: JSON list of font names to drop (see utils.pretraining.PairedImageData)
PROJ_HIDDEN = int(_os.environ.get("PROJ_HIDDEN", "2048"))
CHECKPOINT_DIR = _os.path.join("checkpoints", "pretrain", RUN_NAME)
MODEL_PATH = _os.path.join(CHECKPOINT_DIR, "checkpoint.pt")
STATE_PATH = _os.path.join(CHECKPOINT_DIR, "train_state.pt")
PROGRESS_PATH = _os.path.join(CHECKPOINT_DIR, "progress.json")

# Geometry (defaults: 64px / patch 8 / d512 -- the original run; override per run):
#   FONT_SIZE=32 CACHE_SUFFIX= PATCH_SIZE=4 EMBED_DIM=256  ->  48px, 144 patches, 256-wide
FONT_SIZE = int(_os.environ.get("FONT_SIZE", "43"))        # imageSize = int(fontSize * 1.5)
CACHE_SUFFIX = _os.environ.get("CACHE_SUFFIX", "_64")      # "" = the original bitmaps/smallimage folders (48px)
PATCH_SIZE = int(_os.environ.get("PATCH_SIZE", "8"))
EMBED_DIM = int(_os.environ.get("EMBED_DIM", "512"))
BATCH_SIZE = int(_os.environ.get("BATCH_SIZE", "64"))
SAVE_EVERY = int(_os.environ.get("SAVE_EVERY", "500"))
TEST_EVERY = 4
GEOMETRY_EVERY = 50   # effective-rank diagnostic on the MoCo queue (see queueGeometry)
VRAM_FRACTION = 0.93  # allocator cap: overflow raises OOM instead of spilling into host RAM


class ProjectedViT(nn.Module):
    """forward(x) -> (backbone CLS feature, projector output). The training loop's `_, outputs = model(x)` therefore
    computes the loss on the projector output. Projector = Linear(d, hidden) - BatchNorm - GELU - Linear(hidden, K),
    the LeVJEPA layout. `.vit` alone is a plain ViT (saved as checkpoint.pt so ViT.load / embedding scripts still work)."""

    def __init__(self, vit, projDim, hidden):
        super().__init__()
        self.vit = vit
        dim = vit.config.embedDim
        self.projector = nn.Sequential(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, projDim))

    def forward(self, x):
        _, c = self.vit(x)
        return c, self.projector(c)


class FeatureRing:
    """Fixed-size GPU ring of the momentum encoder's backbone features (+ font ids), for the diagnostic on the space
    that actually gets embedded when a projector is in use."""

    def __init__(self, size, dim):
        self.x = torch.zeros(size, dim, device=device)
        self.f = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr, self.size = 0, size

    @torch.no_grad()
    def add(self, x, f):
        slots = torch.arange(self.ptr, self.ptr + len(x), device=device) % self.size
        self.x[slots] = x.detach()
        self.f[slots] = f.to(device)
        self.ptr = (self.ptr + len(x)) % self.size

    def get(self):
        valid = self.f != -1
        return self.x[valid], None, self.f[valid]


class RandomStreamSampler(Sampler):
    """Uniform sampling WITH replacement, drawn in chunks. torch's RandomSampler builds
    randperm(n).tolist() -- a multi-GB Python int list for this ~116M-pair train set, for an
    epoch this run will never finish."""

    def __init__(self, n, chunk=200_000):
        self.n = n
        self.chunk = chunk

    def __len__(self):
        return self.n

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(int(time.time() * 1000) % (2 ** 31))
        remaining = self.n
        while remaining > 0:
            m = min(self.chunk, remaining)
            yield from torch.randint(0, self.n, (m,), generator=generator).tolist()
            remaining -= m


class ThreadPrefetcher:
    """Runs the (CPU-bound, single-process) data pipeline in a background thread so batch loading
    overlaps GPU compute; adds no extra process memory, unlike DataLoader workers on Windows."""

    def __init__(self, iterable, depth=3):
        self.iterable = iterable
        self.depth = depth

    def __iter__(self):
        buffer = queue.Queue(maxsize=self.depth)
        done = object()

        def produce():
            try:
                for item in self.iterable:
                    buffer.put(item)
            except BaseException as error:  # surfaced in the consumer
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


@torch.no_grad()
def queueGeometry(trainQueue, withNeighbors, prefix="Queue"):
    """Effective dimension (participation ratio = tr(C)^2 / ||C||_F^2, no eigendecomposition) of the stored keys,
    plus mean nearest-OTHER-FONT cosine on a 1024-key subsample. ~9 GFLOP / ~17 GFLOP: a few ms on this GPU.
    Keys are L2-normalised first (the queue already stores normalised keys; the backbone ring does not)."""
    keys, _, families = trainQueue.get()
    if len(keys) < 4096:
        return {}
    keys = nn.functional.normalize(keys.to(device), dim=-1)
    centered = keys - keys.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (len(keys) - 1)
    out = {f"{prefix} PR": (torch.trace(cov) ** 2 / (cov * cov).sum()).item()}
    if withNeighbors:
        families = families.to(device)
        pick = torch.randperm(len(keys), device=device)[:1024]
        sims = keys[pick] @ keys.T
        sims = sims.masked_fill(families[pick].unsqueeze(1) == families.unsqueeze(0), -1.0)
        out[f"{prefix} NN cosine"] = sims.max(dim=1).values.mean().item()
    return out


def endless(loader):
    while True:
        for batch in loader:
            yield batch


def vitOf(model):
    return model.vit if PROJ_DIM else model


def atomicSave(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    _os.replace(tmp, path)


def saveState(config, model, optimizer, momentumModel, total, runId):
    # model is a ProjectedViT (PROJ_DIM>0) or a bare ViT; checkpoint.pt is always a plain ViT state dict
    original = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        atomicSave(vitOf(model).state_dict(), MODEL_PATH)
        config.save(_os.path.join(CHECKPOINT_DIR, "config.json"))
        atomicSave({"step": total, "optimizer": optimizer.state_dict(),
                    "momentum": momentumModel.model.state_dict(), "wandbId": runId,
                    "projector": model.projector.state_dict() if PROJ_DIM else None, "projDim": PROJ_DIM}, STATE_PATH)
        with open(PROGRESS_PATH + ".tmp", "w") as f:
            json.dump({"step": total, "time": time.time()}, f)
        _os.replace(PROGRESS_PATH + ".tmp", PROGRESS_PATH)
    finally:
        signal.signal(signal.SIGINT, original)


def verifyDataset(dataset, config):
    """Fail fast if the images are not really the new resolution from the new cache."""
    imageSize = config.model.imageSize
    assert int(config.dataset.fontSize * 1.5) == imageSize, "dataset.fontSize and model.imageSize disagree"
    rng = np.random.RandomState(0)
    sample = rng.choice(len(dataset.paths), 20000, replace=False)
    if CACHE_SUFFIX:
        stale = [dataset.paths[i] for i in sample if CACHE_SUFFIX not in dataset.paths[i]]
    else:  # default folders: any path inside a size-suffixed folder would be a different resolution
        stale = [dataset.paths[i] for i in sample if "bitmaps_" in dataset.paths[i] or "smallimage_" in dataset.paths[i]]
    assert not stale, f"{len(stale)} sampled paths are outside the {CACHE_SUFFIX!r} cache, e.g. {stale[:2]}"
    for i in sample[:200]:
        _, image = loadImage(dataset.paths[i])
        assert image is not None and image.shape == (imageSize, imageSize), \
            f"{dataset.paths[i]} has shape {None if image is None else image.shape}, expected {imageSize}x{imageSize}"
    print(f"verified: {len(dataset.paths)} glyphs, {len(dataset.fonts)} fonts, {dataset.totalPairs} pairs; "
          f"200/200 sampled bitmaps are {imageSize}x{imageSize} from {CACHE_SUFFIX!r} folders", flush=True)


def trainModel(config):
    dataset = PairedImageData(config.dataset)
    verifyDataset(dataset, config)

    vit = ViT(config.model).to(device)
    model = ProjectedViT(vit, PROJ_DIM, PROJ_HIDDEN).to(device) if PROJ_DIM else vit
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningRate)

    total, runId = 0, None
    state = None
    fullState = False  # optimizer/projector/EMA state compatible with this model
    if _os.path.exists(MODEL_PATH) and _os.path.exists(STATE_PATH):
        vit.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        state = torch.load(STATE_PATH, map_location=device, weights_only=False)
        if PROJ_DIM:
            model.projector.load_state_dict(state["projector"])
        optimizer.load_state_dict(state["optimizer"])
        total, runId, fullState = state["step"], state["wandbId"], True
        print(f"RESUMING from step {total} (wandb run {runId})", flush=True)
    elif INIT_FROM:
        # branch: encoder weights from another run, fresh step counter and wandb run. If that run had no projector
        # (or a different one) the projector, optimizer and EMA start fresh; the EMA is a copy of the loaded model.
        vit.load_state_dict(torch.load(_os.path.join(INIT_FROM, "checkpoint.pt"), map_location=device, weights_only=True))
        state = torch.load(_os.path.join(INIT_FROM, "train_state.pt"), map_location=device, weights_only=False)
        if state.get("projDim", 0) == PROJ_DIM:
            if PROJ_DIM:
                model.projector.load_state_dict(state["projector"])
            optimizer.load_state_dict(state["optimizer"])
            fullState = True
        print(f"BRANCHING from {INIT_FROM} (its step {state['step']}), alpha={ALPHA}, projDim={PROJ_DIM}, "
              f"projector/optimizer/EMA {'loaded' if fullState else 'fresh'}", flush=True)

    momentumModel = MomentumEncoder(model, momentum=config.momentum if "momentum" in config else 0.999)
    if fullState:
        momentumModel.model.load_state_dict(state["momentum"])

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters, device={device}", flush=True)

    queueDim = PROJ_DIM if PROJ_DIM else config.model.embedDim
    trainQueue = MoCoQueue(dim=queueDim, size=config.queueSize)
    testQueue = MoCoQueue(dim=queueDim, size=config.queueSize)
    backboneRing = FeatureRing(config.queueSize, config.model.embedDim)
    objective = EmbeddingLoss(temperature=0.04, alpha=ALPHA, sigregVariant=SIGREG)

    train, test = PairedImageData.split(dataset, config)
    trainLoader = DataLoader(train, batch_size=config.batchSize, sampler=RandomStreamSampler(len(train)),
                             collate_fn=dataset.collate)
    testLoader = DataLoader(test, batch_size=config.batchSize, sampler=RandomStreamSampler(len(test)),
                            collate_fn=dataset.collate)
    testIter = iter(ThreadPrefetcher(endless(testLoader), depth=2))

    run = wandb.init(entity="dylanberndt123-missouri-state-university", project="Briefcase",
                     name=RUN_NAME, config={**config.serialize(), "alpha": ALPHA, "initFrom": INIT_FROM, "projDim": PROJ_DIM, "projHidden": PROJ_HIDDEN, "sigreg": SIGREG}, id=runId, resume="allow")

    try:
        for epoch in range(config.epochs):
            progress = 0
            for inputsX, inputsY, info, ids in ThreadPrefetcher(trainLoader, depth=3):
                assert inputsX.shape[1] == config.model.imageSize and inputsX.shape[2] == config.model.imageSize, f"batch image size {tuple(inputsX.shape)}"
                model.train()
                optimizer.zero_grad()

                _, outputsX = model(inputsX.to(device))
                _, outputsY = model(inputsY.to(device))
                loss = objective(outputsX, outputsY, ids.to(device), queue=trainQueue, enqueue=False)

                trainLoss = loss["total"].detach().item()
                if not np.isfinite(trainLoss):
                    raise FloatingPointError(f"non-finite loss at step {total}: {trainLoss}")

                loss["total"].backward()
                optimizer.step()
                momentumModel.update(model)

                with torch.no_grad():
                    featX, keysX = momentumModel(inputsX.to(device))
                    _, keysY = momentumModel(inputsY.to(device))
                trainQueue.enqueue(keysX, keysY, ids.to(device))
                if PROJ_DIM:
                    backboneRing.add(featX, ids)

                logged = {"Step": total,
                          "Train Loss": trainLoss,
                          "Train InfoNCE": loss["info"].detach().item(),
                          "Train SigREG": loss["sig"].detach().item(),
                          "Train Perplexity": torch.exp(loss["info"].detach()).item(),
                          "Train Recall@1": recallAtK(outputsX, outputsY, k=1, families=ids.to(device), queue=trainQueue),
                          "Train Recall@10": recallAtK(outputsX, outputsY, k=10, families=ids.to(device), queue=trainQueue)}
                testLoss = float("nan")

                if total % TEST_EVERY == 0:
                    with torch.no_grad():
                        model.eval()
                        inputsX1, inputsY1, info1, ids1 = next(testIter)
                        _, outputsX1 = model(inputsX1.to(device))
                        _, outputsY1 = model(inputsY1.to(device))
                        loss1 = objective(outputsX1, outputsY1, ids1.to(device), queue=testQueue, enqueue=False)
                        _, keysX1 = momentumModel(inputsX1.to(device))
                        _, keysY1 = momentumModel(inputsY1.to(device))
                        testQueue.enqueue(keysX1, keysY1, ids1.to(device))
                        testLoss = loss1["total"].detach().item()
                        logged.update({"Test Loss": testLoss,
                                       "Test InfoNCE": loss1["info"].detach().item(),
                                       "Test SigREG": loss1["sig"].detach().item(),
                                       "Test Perplexity": torch.exp(loss1["info"].detach()).item(),
                                       "Test Recall@1": recallAtK(outputsX1, outputsY1, k=1, families=ids1.to(device), queue=testQueue),
                                       "Test Recall@10": recallAtK(outputsX1, outputsY1, k=10, families=ids1.to(device), queue=testQueue)})

                if total % GEOMETRY_EVERY == 0:
                    withNeighbors = total % (GEOMETRY_EVERY * 5) == 0
                    logged.update(queueGeometry(trainQueue, withNeighbors, "Loss-space"))
                    if PROJ_DIM:
                        logged.update(queueGeometry(backboneRing, withNeighbors, "Backbone"))

                run.log(logged)

                progress += 1
                total += 1
                if total % 20 == 0:
                    print(f"\r{epoch + 1} | step {total} | Train Loss: {trainLoss:.2f} | Test Loss: {testLoss:.2f} | "
                          f"SigREG: {logged['Train SigREG']:.3f} | R@1: {logged['Train Recall@1']:.3f}",
                          end="", flush=True)

                if total % SAVE_EVERY == 0:
                    saveState(config, model, optimizer, momentumModel, total, run.id)

    except KeyboardInterrupt:
        pass
    saveState(config, model, optimizer, momentumModel, total, run.id)


def main():
    config = Config().load(_os.path.join("configs", "vit.json"))
    # queueSize stays at configs/vit.json's 16384 (= best/latest): the fix here is the loss.
    config.dataset.fontSize = FONT_SIZE
    config.dataset.cacheSuffix = CACHE_SUFFIX
    if EXCLUDE_FONTS_FILE:
        config.dataset.excludeFontsFile = EXCLUDE_FONTS_FILE
    config.model.imageSize = int(FONT_SIZE * 1.5)
    config.model.patchSize = PATCH_SIZE
    config.model.embedDim = EMBED_DIM
    config.batchSize = BATCH_SIZE

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)

    print(f"{RUN_NAME}: queueSize={config.queueSize} batchSize={config.batchSize} "
          f"imageSize={config.model.imageSize} patchSize={config.model.patchSize} "
          f"patches={(config.model.imageSize // config.model.patchSize) ** 2} embedDim={config.model.embedDim} "
          f"projDim={PROJ_DIM} alpha={ALPHA} sigreg={SIGREG} "
          f"checkpointDir={CHECKPOINT_DIR}", flush=True)
    trainModel(config)


if __name__ == "__main__":
    main()
