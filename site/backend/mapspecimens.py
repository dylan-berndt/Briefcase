"""The pictures the maps show on hover: each map point's own name set in its own font (built by site/tools/mapSpecimens.py).

    data/maps/<map>.pack         the images back to back
    data/maps/<map>.index.json   {"count": points, "specimens": [[offset, length] or null per point, in point order]}

A map with no index here simply has none, and a point with a null entry has none; the page falls back to the plain label.
"""

import hashlib
import json
import os

import numpy as np


class MapSpecimens:
    def __init__(self, directory):
        self.maps = {}
        if not os.path.isdir(directory):
            return
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".index.json"):
                continue
            stem = name[:-len(".index.json")]
            packPath = os.path.join(directory, stem + ".pack")
            if not os.path.exists(packPath):
                continue
            with open(os.path.join(directory, name), "rb") as file:
                raw = file.read()
            specimens = json.loads(raw)["specimens"]
            pack = np.memmap(packPath, dtype=np.uint8, mode="r") if os.path.getsize(packPath) else np.empty(0, np.uint8)
            for entry in specimens:
                if entry is not None and entry[0] + entry[1] > len(pack):
                    # a git-lfs pointer left behind by a checkout without `git lfs pull` is tiny, and fails here
                    raise ValueError(f"{stem}.pack is {len(pack)} bytes, too short for its index "
                                     "(truncated copy, or an unpulled git-lfs pointer?)")
            self.maps[stem] = {"specimens": specimens, "pack": pack, "version": hashlib.sha256(raw).hexdigest()[:12]}

    def get(self, name, point):
        """(image bytes, version) for a point of a map, or None when there is none."""
        entry = self.maps.get(name)
        if entry is None or not 0 <= point < len(entry["specimens"]) or entry["specimens"][point] is None:
            return None
        offset, length = entry["specimens"][point]
        return entry["pack"][offset:offset + length].tobytes(), entry["version"]
