"""
Embeds fonts with a LeVJEPA encoder: the whole font in one pass (every patch of every present glyph, no token dropping),
CLS output after the final LayerNorm (the backbone feature, not the projector). Same embedding as probe.py --mode full.

API (import from experiments/levjepa):
    encoder = loadEncoder("checkpoints/pretrain/levjepa-d256-e200-snapshots/step38500")
    vector  = embedFont(encoder, glyphs, present)            # glyphs uint8 [52, 48, 48], present bool [52] -> [D]
    matrix  = embedFonts(encoder, glyphs, present, batch=32)  # [N, 52, 48, 48], [N, 52] -> [N, D]

Glyph layout: index 0-25 = a-z, 26-51 = A-Z, 48px, glyph = 255 on 0, rendered by build_glyph_cache.py's standardized
(MyFonts-style) pipeline. Fonts rendered any other way are out of distribution.

CLI: embed every font in dataset/levjepa and write
    embeddings/<name>.npz   names, sources, X float32 [N, D]      (all rows, duplicate names kept)
    embeddings/<name>.json  {font name: vector}                    (repo convention; a name that appears in two sources
                                                                    gets " (<source>)" appended on the second one)

    python -u experiments/levjepa/embed.py --ckpt checkpoints/pretrain/levjepa-d256-e200-snapshots/step38500
"""
import argparse
import json
import os
import sys
import time

_scriptDir = os.path.dirname(os.path.abspath(__file__))
_repoRoot = os.path.dirname(os.path.dirname(_scriptDir))
for _p in (_scriptDir, _repoRoot):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import train as levjepa

GRID, PATCH, PER_GLYPH = levjepa.GRID, levjepa.PATCH, levjepa.PER_GLYPH


def loadEncoder(ckptDir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(ckptDir, "config.json")) as f:
        m = json.load(f)["model"]
    enc = levjepa.Encoder(m["dim"], m["depth"], m["heads"], m["mlpRatio"])
    enc.load_state_dict(torch.load(os.path.join(ckptDir, "checkpoint.pt"), map_location="cpu", weights_only=True))
    return enc.to(device).eval()


def _tokens(x, letters):
    """x float [B, G, 48, 48], letters long [G] -> tokens [B, G*36, 64], coords [B, G*36, 3] (letter, row, col)."""
    B, G = x.shape[:2]
    p = x.view(B, G, GRID, PATCH, GRID, PATCH).permute(0, 1, 2, 4, 3, 5).reshape(B, G * PER_GLYPH, PATCH * PATCH)
    pos = torch.arange(PER_GLYPH, device=x.device).repeat(G)
    coords = torch.stack([letters.repeat_interleave(PER_GLYPH).float(), (pos // GRID).float(), (pos % GRID).float()], -1)
    return p, coords.unsqueeze(0).expand(B, -1, -1)


@torch.no_grad()
def embedFonts(encoder, glyphs, present, batch=32):
    """glyphs uint8 [N, 52, 48, 48] (array or memmap), present bool [N, 52] -> float32 [N, D]. Fonts with all 52 glyphs
    are batched; fonts with missing glyphs are embedded one at a time from their present glyphs only."""
    device = next(encoder.parameters()).device
    present = np.asarray(present, bool)
    D = encoder.norm.normalized_shape[0]
    out = np.zeros((len(present), D), np.float32)
    full = np.where(present.all(1))[0]
    letters = torch.arange(present.shape[1], device=device)
    for s in range(0, len(full), batch):
        ids = full[s:s + batch]
        x = torch.from_numpy(np.ascontiguousarray(glyphs[ids])).to(device).float() / 255.0
        out[ids] = encoder(*_tokens(x, letters)).float().cpu().numpy()
    for i in np.where(~present.all(1) & present.any(1))[0]:
        keep = np.where(present[i])[0]
        x = torch.from_numpy(np.ascontiguousarray(glyphs[i][keep]))[None].to(device).float() / 255.0
        out[i] = encoder(*_tokens(x, torch.as_tensor(keep, device=device))).float().cpu().numpy()[0]
    return out


def embedFont(encoder, glyphs, present):
    return embedFonts(encoder, np.asarray(glyphs)[None], np.asarray(present, bool)[None], batch=1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=os.path.join("dataset", "levjepa"))
    p.add_argument("--name", default=None, help="output name under embeddings/ (default levjepa_<run>_<step>)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--vramFraction", type=float, default=0.5)
    p.add_argument("--noJson", action="store_true")
    args = p.parse_args()
    os.chdir(_repoRoot)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.vramFraction)
    ck = os.path.normpath(args.ckpt)
    name = args.name or f"levjepa_{os.path.basename(os.path.dirname(ck)).replace('-snapshots', '')}_{os.path.basename(ck)}"

    glyphs = np.load(os.path.join(args.data, "glyphs.npy"), mmap_mode="r")
    present = np.load(os.path.join(args.data, "present.npy"))
    with open(os.path.join(args.data, "meta.json"), encoding="utf8") as f:
        meta = json.load(f)
    n = args.limit or len(present)
    started = time.time()
    X = embedFonts(loadEncoder(args.ckpt, device), glyphs[:n], present[:n], args.batch)
    names, sources = meta["names"][:n], meta["sources"][:n]
    empty = int((~present[:n].any(1)).sum())

    os.makedirs("embeddings", exist_ok=True)
    np.savez(os.path.join("embeddings", name + ".npz"), names=np.array(names), sources=np.array(sources), X=X)
    if not args.noJson:
        out = {}
        for nm, src, v in zip(names, sources, X):
            key = nm if nm not in out else f"{nm} ({src})"
            out[key] = v.tolist()
        with open(os.path.join("embeddings", name + ".json"), "w") as f:
            json.dump(out, f)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
    print(f"{n} fonts ({empty} with no glyphs, left as zeros) -> embeddings/{name}.npz"
          f"{'' if args.noJson else ' + .json'} in {time.time() - started:.0f}s, peak GPU {peak:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
