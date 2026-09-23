"""
Participation ratio + nearest-neighbor cosine (same functions as geometry_comparison.py) for
several visual embedding files, restricted to the fonts they share.

    python experiments/embedding-geometry/compare_visual_geometry.py all all_strong64_step38500
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_comparison import participationRatio, nearestNeighborCosine, REPO_ROOT

files = {n: json.load(open(os.path.join(REPO_ROOT, "embeddings", n + ".json"))) for n in sys.argv[1:]}
shared = sorted(set.intersection(*[set(d) for d in files.values()]))
print(f"shared fonts: {len(shared)}  ({', '.join(f'{n}: {len(d)}' for n, d in files.items())})")
for n, d in files.items():
    for label, keys in (("all own fonts", list(d)), ("shared fonts", shared)):
        v = np.array([d[k] for k in keys], dtype=np.float64)
        pr, d95 = participationRatio(v)
        nn = nearestNeighborCosine(v)
        norms = np.linalg.norm(v, axis=1)
        print(f"{n:28s} {label:14s} n={len(v):6d} PR={pr:6.1f} dims95={d95:4d} NNcos={nn}  norm[min,max]=[{norms.min():.2f},{norms.max():.2f}]")
