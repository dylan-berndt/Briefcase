"""A synthetic bundle in the real format, for developing and testing the site without the real model output.

Every font gets a few planted tag groups, so a query for a group has known correct answers. The fonts are not real:
names are "Fake 0001" and so on, and the specimens are placeholders.

    python site/tools/fakeBundle.py site/backend/data-dev          # then BUNDLE_DIR=site/backend/data-dev
"""

import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bundleWriter import writeBundle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# tags that are neither merged into a canonical nor dropped, so only searchable through the extraTags fallback
EXTRA_TAGS = ["zebra-stripe", "moss", "%E3%81%82%E3%81%84"]


def placeholder(label):
    image = Image.new("RGBA", (640, 180), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((24, 80), label, fill=(232, 230, 224, 255))
    out = io.BytesIO()
    image.save(out, format="WEBP")
    return out.getvalue()


def makeFakeBundle(directory, numFonts=300, seed=0):
    """Returns {fontKey: [canonical groups planted in it]}."""
    rng = np.random.RandomState(seed)
    with open(os.path.join(REPO, "configs", "tagVocabulary.json")) as file:
        canonical = json.load(file)["canonical"]
    vocab = sorted({m for entry in canonical.values() for m in entry["members"]}) + EXTRA_TAGS
    row = {tag: i for i, tag in enumerate(vocab)}
    groups = sorted(canonical)

    logits = rng.normal(-5.0, 0.7, size=(numFonts, len(vocab)))
    fonts, planted = [], {}
    for i in range(numFonts):
        chosen = list(rng.choice(groups, size=rng.randint(1, 4), replace=False))
        if i % 10 == 0:
            chosen.append("zebra-stripe")
        for name in chosen:
            members = canonical[name]["members"] if name in canonical else [name]
            logits[i, row[members[0]]] = rng.normal(5.0, 0.5)
        source = "google" if i % 3 else "dafont"
        key = f"{source}:Fake {i:04d}"
        planted[key] = chosen
        fonts.append({
            "key": key,
            "name": f"Fake {i:04d}",
            "source": source,
            "url": (f"https://fonts.google.com/specimen/Fake+{i:04d}" if source == "google"
                    else f"https://www.dafont.com/fake-{i:04d}.font"),
            "creator": None if source == "google" else f"Designer {i % 7}",
        })

    writeBundle(directory, fonts, vocab, logits, [placeholder(f["name"]) for f in fonts],
                model={"adapter": "fake", "path": None})
    return planted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--numFonts", type=int, default=300)
    args = parser.parse_args()
    makeFakeBundle(args.directory, args.numFonts)
    print(f"wrote a fake bundle of {args.numFonts} fonts to {args.directory}")
