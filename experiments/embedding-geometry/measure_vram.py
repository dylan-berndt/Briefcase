"""
Peak VRAM + step-time measurement for candidate (imageSize, patchSize, batch)
configs, using synthetic data (no dataset loading). IMPORTANT: this machine's
driver silently spills GPU allocations into system RAM instead of raising CUDA
OOM (earlier measurements showed 7-15 GB "peaks" on a 6.4 GB card that simply
succeeded), so a hard per-process allocator cap is set here -- a config that
exceeds it reports OOM instead of quietly thrashing host RAM.

    python -u experiments/embedding-geometry/measure_vram.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)

import statistics
import time

import torch

from utils.config import Config
from utils.training import EmbeddingLoss, MoCoQueue, MomentumEncoder
from utils.vit import ViT

FRACTION = 0.90  # of total VRAM -- allocator raises OOM instead of spilling to host RAM
CONFIGS = [
    (66, 8, 128), (66, 8, 96), (66, 8, 64),
    (96, 8, 64), (96, 8, 48), (96, 8, 32), (96, 8, 24),
]


def measure(imageSize, patchSize, batchSize, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    modelConfig = Config(layers=12, imageSize=imageSize, patchSize=patchSize, embedDim=512, heads=8)
    model = ViT(modelConfig).to(device)
    momentumModel = MomentumEncoder(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    objective = EmbeddingLoss(temperature=0.04, alpha=0.1)
    queue = MoCoQueue(dim=512, size=16384)
    # fill the queue so InfoNCE really sees B+16384 keys, as in real training
    queue.enqueue(torch.randn(16384, 512, device=device), torch.randn(16384, 512, device=device),
                  torch.randint(0, 40000, (16384,), device=device))

    x = torch.randn(batchSize, imageSize, imageSize, 1, device=device)
    y = torch.randn(batchSize, imageSize, imageSize, 1, device=device)
    ids = torch.randint(0, 40000, (batchSize,), device=device)

    times = []
    for step in range(7):
        torch.cuda.synchronize()
        start = time.time()
        optimizer.zero_grad()
        _, outputsX = model(x)
        _, outputsY = model(y)
        loss = objective(outputsX, outputsY, ids, queue=queue, enqueue=False)
        loss["total"].backward()
        optimizer.step()
        with torch.no_grad():
            momentumModel.update(model)
            _, kx = momentumModel(x)
            _, ky = momentumModel(y)
            model.eval()
            model(x)
            model.train()
        queue.enqueue(kx, ky, ids)
        torch.cuda.synchronize()
        if step >= 2:
            times.append(time.time() - start)

    allocated = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    return allocated, reserved, statistics.median(times)


def main():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(FRACTION)
    print(f"total VRAM {total / 1e9:.2f} GB, allocator cap {FRACTION * total / 1e9:.2f} GB", flush=True)
    for imageSize, patchSize, batchSize in CONFIGS:
        patches = (imageSize // patchSize) ** 2
        try:
            allocated, reserved, seconds = measure(imageSize, patchSize, batchSize)
            print(f"image={imageSize} patch={patchSize} patches={patches} batch={batchSize}: "
                  f"allocated={allocated:.2f}GB reserved={reserved:.2f}GB gpuStep={seconds * 1000:.0f}ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"image={imageSize} patch={patchSize} patches={patches} batch={batchSize}: OOM (over cap)", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
