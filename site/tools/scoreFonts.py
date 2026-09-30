"""Step 2 of building the search bundle: run a tagger over every font. The only step that needs the model and a GPU,
and the only one to redo when the model changes.

Fonts are rendered the way the MyFonts training images are (tight crop, scale to fit; experiments/critical-review/
render.py), which is what finetuneTags.py trained on.

    python site/tools/scoreFonts.py --fonts build/fonts.json --out build/scores.npz
    python site/tools/scoreFonts.py --adapter finetuneTags --model checkpoints/retrieval/finetuneTags/<run> ...

--model is an adapter-specific path (for finetuneTags: a run directory, or the directory of runs, which picks the
newest finished one). Add adapters in taggers.py.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "experiments", "critical-review"))

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def renderGlyphs(task):
    from render import myfontsStyleFromFont
    from fontList import hasLowercase
    path, fontSize = task
    # PIL draws the .notdef box for a character the font lacks, which would be scored as if it were a letter
    if not hasLowercase(path):
        return None
    try:
        glyphs = [myfontsStyleFromFont(path, letter, fontSize=fontSize) for letter in LETTERS]
    except Exception:
        return None
    if any(g is None for g in glyphs):
        return None
    return np.round(np.stack(glyphs) * 255).astype(np.uint8)


def scoreFonts(fonts, tagger, batchSize=64, workers=None):
    """Returns (keys that were scored, logits [n, numTags] float32, number of fonts that would not render)."""
    keys, logits, failed = [], [], 0
    pending, pendingKeys = [], []

    def flush():
        if pending:
            logits.append(tagger.logits(np.stack(pending)))
            keys.extend(pendingKeys)
            pending.clear()
            pendingKeys.clear()

    tasks = [(font["path"], tagger.fontSize) for font in fonts]
    with Pool(workers) as pool:
        for i, glyphs in enumerate(pool.imap(renderGlyphs, tasks, chunksize=16)):
            if glyphs is None:
                failed += 1
            else:
                pending.append(glyphs)
                pendingKeys.append(fonts[i]["key"])
                if len(pending) == batchSize:
                    flush()
            if i % 500 == 0:
                print(f"\r{i + 1}/{len(fonts)} fonts", end="", flush=True)
        flush()
    print()
    return keys, np.concatenate(logits) if logits else np.zeros((0, len(tagger.vocab)), np.float32), failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", default=os.path.join("build", "fonts.json"))
    parser.add_argument("--adapter", default="finetuneTags")
    parser.add_argument("--model", default=os.path.join("checkpoints", "retrieval", "finetuneTags"))
    parser.add_argument("--out", default=os.path.join("build", "scores.npz"))
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from taggers import loadTagger
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.fonts) as file:
        fonts = json.load(file)
    tagger = loadTagger(args.adapter, args.model, device)
    run = getattr(tagger, "run", args.model)
    print(f"{args.adapter} from {run} on {device}: {len(tagger.vocab)} tags, {len(fonts)} fonts")

    keys, logits, failed = scoreFonts(fonts, tagger, args.batchSize, args.workers)
    print(f"scored {len(keys)} fonts, {failed} would not render")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = {"adapter": args.adapter, "path": run, "device": device, "failedToRender": failed}
    np.savez(args.out, keys=np.array(keys), logits=logits.astype(np.float16), vocab=json.dumps(tagger.vocab),
             meta=json.dumps(meta))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
