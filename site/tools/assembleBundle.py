"""Step 3 of building the search bundle: render a specimen for every scored font and write the bundle the server
reads (site/backend/data by default). Needs no model, so it runs anywhere.

    python site/tools/assembleBundle.py --fonts build/fonts.json --scores build/scores.npz --out site/backend/data

Commit the result through git-lfs (see .gitattributes). The server refuses to start on a bundle that is incomplete.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from bundleWriter import writeBundle  # noqa: E402
from specimens import renderSpecimen  # noqa: E402


def assemble(fonts, scores, directory, workers=None):
    """fonts: entries from listFonts; scores: the npz scoreFonts wrote. Fonts without a score or without a
    specimen are dropped. Returns the manifest."""
    row = {key: i for i, key in enumerate(scores["keys"])}
    kept = [f for f in fonts if f["key"] in row]

    with Pool(workers) as pool:
        images = pool.map(renderSpecimen, [f["path"] for f in kept], chunksize=16)
    drawable = [i for i, image in enumerate(images) if image is not None]
    print(f"{len(kept)} scored fonts, {len(kept) - len(drawable)} have no specimen characters")
    kept = [kept[i] for i in drawable]

    logits = scores["logits"][[row[f["key"]] for f in kept]]
    shipped = [{k: f[k] for k in ("key", "name", "source", "url", "creator")} for f in kept]
    return writeBundle(directory, shipped, json.loads(str(scores["vocab"])), logits,
                       [images[i] for i in drawable], model=json.loads(str(scores["meta"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", default=os.path.join("build", "fonts.json"))
    parser.add_argument("--scores", default=os.path.join("build", "scores.npz"))
    parser.add_argument("--out", default=os.path.join("site", "backend", "data"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    with open(args.fonts) as file:
        fonts = json.load(file)
    manifest = assemble(fonts, np.load(args.scores), args.out, args.workers)

    megabytes = {name: round(f["bytes"] / 1e6, 1) for name, f in manifest["files"].items()}
    print(f"wrote {manifest['numFonts']} fonts x {manifest['numTags']} tags to {args.out}: {megabytes} MB")


if __name__ == "__main__":
    main()
