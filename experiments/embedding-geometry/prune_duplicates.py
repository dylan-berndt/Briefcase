"""
For a given font embedding space: build the graph where fonts are connected if their cosine
similarity exceeds a threshold, then dedup by keeping ONE representative per connected component
(the natural generalization of "if A~B and B~C are both near-duplicate, keep just one of the three,
not two") -- and report what fraction of the corpus that prunes, swept over several thresholds.

Computes the full pairwise similarity in row-chunks on GPU (no NxN matrix ever materialized at once),
collects every (i,j) pair above the LOWEST threshold in the sweep once, then re-filters/re-clusters
that same edge list per threshold (cheap -- no repeated GPU work).

    python experiments/embedding-geometry/prune_duplicates.py <embeddingsName> [<embeddingsName> ...]
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)

import json

import numpy as np
import torch

CHUNK = 4000
LOW_THRESHOLD = 0.90        # collect every pair above this once; sweep re-filters from here
THRESHOLDS = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def collectEdges(vecs, device):
    n = vecs.shape[0]
    v = torch.tensor(vecs, dtype=torch.float32, device=device)
    v = torch.nn.functional.normalize(v, dim=-1)
    allI, allJ, allS = [], [], []
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        block = v[start:end] @ v.T  # [chunk, n]
        # only keep j > global_i to avoid double-counting/self-matches
        rowIdx = torch.arange(start, end, device=device).unsqueeze(1)
        colIdx = torch.arange(n, device=device).unsqueeze(0)
        mask = (block > LOW_THRESHOLD) & (colIdx > rowIdx)
        ii, jj = torch.nonzero(mask, as_tuple=True)
        if len(ii):
            allI.append((ii + start).cpu())
            allJ.append(jj.cpu())
            allS.append(block[ii, jj].cpu())
        print(f"\r  rows {end}/{n}, edges so far: {sum(len(x) for x in allI)}", end="", flush=True)
    print()
    if not allI:
        return np.array([]), np.array([]), np.array([])
    return (torch.cat(allI).numpy(), torch.cat(allJ).numpy(), torch.cat(allS).numpy())


def sweep(n, edgeI, edgeJ, edgeS, names):
    print(f"{'threshold':>10s} {'edges':>10s} {'nodes-touched':>14s} {'components':>11s} {'pruned':>8s} {'pct-of-corpus':>14s}")
    examples = {}
    for t in THRESHOLDS:
        keep = edgeS >= t
        i, j = edgeI[keep], edgeJ[keep]
        uf = UnionFind(n)
        for a, b in zip(i, j):
            uf.union(int(a), int(b))
        touched = set(i.tolist()) | set(j.tolist())
        roots = set(uf.find(x) for x in touched)
        pruned = len(touched) - len(roots)
        print(f"{t:10.3f} {len(i):10d} {len(touched):14d} {len(roots):11d} {pruned:8d} {pruned/n*100:13.2f}%")
        if t in (0.95, 0.99) and len(i) > 0:
            examples[t] = (i[0], j[0], edgeS[keep][0])
    for t, (a, b, s) in examples.items():
        print(f"  example pair at threshold {t}: {names[a]!r} <-> {names[b]!r}  cos={s:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for embName in _sys.argv[1:]:
        with open(_os.path.join("embeddings", f"{embName}.json")) as f:
            data = json.load(f)
        names = sorted(data)
        vecs = np.array([data[n] for n in names], dtype=np.float64)
        print(f"\n=== {embName} ({len(names)} fonts, dim {vecs.shape[1]}) ===", flush=True)
        edgeI, edgeJ, edgeS = collectEdges(vecs, device)
        sweep(len(names), edgeI, edgeJ, edgeS, names)


if __name__ == "__main__":
    main()
