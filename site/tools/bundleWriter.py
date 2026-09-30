"""Writes the serving bundle that site/backend/bundle.py reads. Shared by assembleBundle.py and the fake bundle
the tests use, so both go through the same format."""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from bundle import FORMAT, sha256  # noqa: E402

LOGIT_LIMIT = 30.0


def writeBundle(directory, fonts, vocab, logits, specimens, mimetype="image/webp", model=None):
    """fonts: [{key, name, source, url, creator}] in row order; logits: [numFonts, numTags] scores from the tagger;
    specimens: one encoded image (bytes) per font. Fills in each font's specimen [offset, length]."""
    logits = np.asarray(logits)
    if logits.shape != (len(fonts), len(vocab)):
        raise ValueError(f"logits {logits.shape} do not match {len(fonts)} fonts x {len(vocab)} tags")
    if len(specimens) != len(fonts):
        raise ValueError("need exactly one specimen per font")
    if len({f["key"] for f in fonts}) != len(fonts):
        raise ValueError("font keys are not unique")

    os.makedirs(directory, exist_ok=True)
    entries, offset = [], 0
    with open(os.path.join(directory, "specimens.pack"), "wb") as pack:
        for font, image in zip(fonts, specimens):
            pack.write(image)
            entries.append({**font, "specimen": [offset, len(image)]})
            offset += len(image)

    with open(os.path.join(directory, "fonts.json"), "w") as file:
        json.dump(entries, file, separators=(",", ":"))
    with open(os.path.join(directory, "vocab.json"), "w") as file:
        json.dump(list(vocab), file, separators=(",", ":"))
    # tag-major, so the server reads one tag's scores for every font as one contiguous row
    np.save(os.path.join(directory, "logits.npy"),
            np.ascontiguousarray(np.clip(logits, -LOGIT_LIMIT, LOGIT_LIMIT).T.astype(np.float16)))

    files = {}
    for name in ("vocab.json", "fonts.json", "logits.npy", "specimens.pack"):
        path = os.path.join(directory, name)
        files[name] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
    manifest = {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numFonts": len(fonts),
        "numTags": len(vocab),
        "specimen": {"mimetype": mimetype},
        "model": model or {},
        "files": files,
    }
    with open(os.path.join(directory, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=1)
    return manifest
