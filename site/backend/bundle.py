"""The serving bundle: everything the site needs to search, built offline by site/tools/assembleBundle.py.

    manifest.json   format version, sizes and checksums, which model produced the scores
    vocab.json      the tagger's tags, in column order of logits.npy
    fonts.json      one entry per font: key, name, source, url, creator, specimen [offset, length]
    logits.npy      float16 [numTags, numFonts]: tagger logits, tag-major so one tag's scores are contiguous
    specimens.pack  every specimen image back to back, located through fonts.json

Nothing here needs torch: the tagger ran offline, the server only reads its outputs.
"""

import hashlib
import json
import os

import numpy as np

FORMAT = 1
FILES = ("vocab.json", "fonts.json", "logits.npy", "specimens.pack")


class BundleError(Exception):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Bundle:
    def __init__(self, directory, verify=False):
        self.directory = directory
        manifestPath = os.path.join(directory, "manifest.json")
        if not os.path.exists(manifestPath):
            raise BundleError(f"no manifest.json in {directory!r}; build the bundle with site/tools/assembleBundle.py")
        with open(manifestPath) as file:
            self.manifest = json.load(file)
        if self.manifest.get("format") != FORMAT:
            raise BundleError(f"bundle format {self.manifest.get('format')!r}, this server reads format {FORMAT}")

        for name in FILES:
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                raise BundleError(f"bundle is missing {name}")
            expected = self.manifest["files"][name]
            # an LFS pointer file left behind by a checkout without `git lfs pull` is tiny, and fails here
            if os.path.getsize(path) != expected["bytes"]:
                raise BundleError(f"{name} is {os.path.getsize(path)} bytes, manifest says {expected['bytes']} "
                                  "(truncated copy, or an unpulled git-lfs pointer?)")
            if verify and sha256(path) != expected["sha256"]:
                raise BundleError(f"{name} does not match its checksum")

        with open(os.path.join(directory, "vocab.json")) as file:
            self.vocab = json.load(file)
        with open(os.path.join(directory, "fonts.json")) as file:
            self.fonts = json.load(file)
        self.logits = np.load(os.path.join(directory, "logits.npy"), mmap_mode="r")
        self.specimens = np.memmap(os.path.join(directory, "specimens.pack"), dtype=np.uint8, mode="r")

        if self.logits.dtype != np.float16 or self.logits.shape != (len(self.vocab), len(self.fonts)):
            raise BundleError(f"logits.npy is {self.logits.dtype}{self.logits.shape}, expected "
                              f"float16 ({len(self.vocab)}, {len(self.fonts)})")
        keys = [font["key"] for font in self.fonts]
        if len(set(keys)) != len(keys):
            raise BundleError("fonts.json has duplicate keys")
        for font in self.fonts:
            offset, length = font["specimen"]
            if offset < 0 or length <= 0 or offset + length > len(self.specimens):
                raise BundleError(f"specimen of {font['key']!r} lies outside specimens.pack")

        self.indexByKey = {key: i for i, key in enumerate(keys)}
        # Changes whenever any file does; used to bust the browser cache on specimen URLs
        self.version = hashlib.sha256(json.dumps(self.manifest["files"], sort_keys=True).encode()).hexdigest()[:12]

    def specimen(self, index):
        offset, length = self.fonts[index]["specimen"]
        return self.specimens[offset:offset + length].tobytes()
