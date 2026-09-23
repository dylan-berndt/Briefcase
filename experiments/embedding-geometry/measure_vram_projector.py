"""
Peak VRAM + GPU step time for the projector training setup (ProjectedViT from retrain_strong_sigreg.py), same synthetic
loop as measure_vram.py: two views through the online model, InfoNCE against a full 16384-key queue + SIGReg, backward,
EMA update, two momentum forwards, an eval-mode forward (test step), queue + backbone-ring updates. Hard allocator cap so
overflow raises OOM instead of spilling into host RAM.

    python -u experiments/embedding-geometry/measure_vram_projector.py <embedDim> <imageSize> <patchSize> <batch> [<batch> ...]
"""
import os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import retrain_strong_sigreg as R
from utils.config import Config
from utils.training import EmbeddingLoss, MoCoQueue, MomentumEncoder
from utils.vit import ViT

FRACTION = 0.90


def measure(embedDim, imageSize, patchSize, batch, alpha=0.5, device="cuda"):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = Config(layers=12, imageSize=imageSize, patchSize=patchSize, embedDim=embedDim, heads=8)
    model = R.ProjectedViT(ViT(cfg), R.PROJ_DIM, R.PROJ_HIDDEN).to(device)
    ema = MomentumEncoder(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    obj = EmbeddingLoss(temperature=0.04, alpha=alpha)
    q = MoCoQueue(dim=R.PROJ_DIM, size=16384)
    q.enqueue(torch.randn(16384, R.PROJ_DIM, device=device), torch.randn(16384, R.PROJ_DIM, device=device),
              torch.randint(0, 40000, (16384,), device=device))
    ring = R.FeatureRing(16384, embedDim)
    x = torch.randn(batch, imageSize, imageSize, 1, device=device); y = torch.randn_like(x)
    ids = torch.randint(0, 40000, (batch,), device=device)
    params = sum(p.numel() for p in model.parameters())
    times = []
    for step in range(8):
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad()
        _, ox = model(x); _, oy = model(y)
        loss = obj(ox, oy, ids, queue=q, enqueue=False)
        loss["total"].backward(); opt.step()
        with torch.no_grad():
            ema.update(model)
            fx, kx = ema(x); _, ky = ema(y)
            model.eval(); model(x); model.train()
            if step % 5 == 0:
                R.queueGeometry(q, True, "Loss-space")
        q.enqueue(kx, ky, ids); ring.add(fx, ids)
        torch.cuda.synchronize()
        if step >= 3: times.append(time.time() - t0)
    return params, torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9, statistics.median(times)


if __name__ == "__main__":
    embedDim, imageSize, patchSize = map(int, sys.argv[1:4])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(FRACTION)
    print(f"total VRAM {total/1e9:.2f} GB, allocator cap {FRACTION*total/1e9:.2f} GB; embedDim={embedDim} image={imageSize} "
          f"patch={patchSize} patches={(imageSize//patchSize)**2} projDim={R.PROJ_DIM}", flush=True)
    for batch in map(int, sys.argv[4:]):
        try:
            params, a, r, s = measure(embedDim, imageSize, patchSize, batch)
            print(f"  batch={batch:4d}: params={params/1e6:.1f}M allocated={a:.2f}GB reserved={r:.2f}GB gpuStep={s*1000:.0f}ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch:4d}: OOM (over cap)", flush=True)
        torch.cuda.empty_cache()
