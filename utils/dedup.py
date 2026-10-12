"""
Near-duplicate fonts across the whole compiled corpus, run after every source's glyphs are cached and loaded.

Each font is represented by DEDUP_LETTERS, each glyph re-centred on its ink centroid and rescaled by its ink bounding
box (centerAndScaleNormalize), concatenated into one 8 x 48 x 48 vector; similarity is the cosine of two such vectors.
Fonts missing a dedup letter can't be compared and are kept. Fonts that draw the same placeholder box for every
letter (they map the letters in their cmap but every glyph is a .notdef-style rectangle) are always dropped.

Rule for a threshold t: visit every pair with s > t in random order; if both fonts are still kept, drop one of the
two at random, except that a MyFonts font is kept over a font from any other source (MyFonts carries the tags the
tagging models train on; the font files of other sources don't). Byte-identical font files already share one key
(standard.fontKey hashes the file), so they collapse before this step.

The pair search is exact (blockwise matmul over the upper triangle, on a CUDA GPU through torch when available):
an SVD projection + faiss range search was tried and its cosine error (0.29 at the 99.99th percentile, 384 dims)
made the candidate set blow up. All pairs above MIN_THRESHOLD are cached, so any t >= MIN_THRESHOLD is instant.

Used by:
  * utils.pretraining.PairedImageData / utils.querying.CombinedQueryData via applyDedup (config key
    "dedupThreshold"), on the bitmap caches from loaders/standard.py and loaders/myfonts.py.
  * experiments/levjepa/build_glyph_cache.py (writes dedup.npz next to glyphs.npy) and train.py (config key
    dataset.dedupThreshold).

Table of thresholds vs. fonts discarded:
    python -m utils.dedup --config configs/vit.json
    python -m utils.dedup --glyphCache dataset/levjepa

No relative imports, so scripts that avoid utils/__init__ (torch, transformers) can load this file by path.
"""
import argparse
import hashlib
import json
import os
import time

import cv2
import numpy as np
from scipy.ndimage import affine_transform

DEDUP_LETTERS = "aegknHQR"
IMAGE_SIZE = 48
MIN_THRESHOLD = 0.90
THRESHOLDS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
PLACEHOLDER_SIMILARITY = 0.999   # mean cosine between a font's own dedup letters; above = one box drawn for all
DEFAULT_CACHE_DIR = os.path.join("dataset", "dedup")


def centerAndScaleNormalize(img, outSize=48):
    """Per-IMAGE normalization (no pairing involved): recenter on the image's own ink centroid and
    rescale so its own ink bounding box fills a fixed fraction of the canvas. This is what makes the
    cheap prefilter shift/scale-INVARIANT -- two accidentally-shifted/scaled copies of the same glyph
    map to nearly the same normalized image, without ever comparing them to each other."""
    mass = img.sum()
    if mass < 1e-6:
        return img
    ys, xs = np.nonzero(img > 0.05)
    if len(xs) == 0:
        return img
    cy, cx = ys.mean(), xs.mean()
    h = max(ys.max() - ys.min(), 1)
    w = max(xs.max() - xs.min(), 1)
    scale = (outSize * 0.6) / max(h, w)   # target ink extent ~60% of canvas, same for every glyph
    matrix = np.eye(2) / scale
    center = np.array([outSize / 2, outSize / 2])
    offset = np.array([cy, cx]) - matrix @ center
    return affine_transform(img, matrix, offset=offset, output_shape=(outSize, outSize), order=1, mode="constant", cval=0.0)


def dedupVector(glyphs):
    """glyphs: the DEDUP_LETTERS images (float, glyph = 1), in order -> [len(DEDUP_LETTERS), 48*48] float32."""
    return np.stack([centerAndScaleNormalize(g.astype(np.float64)).reshape(-1) for g in glyphs]).astype(np.float32)


def interLetterSimilarity(perLetter):
    """perLetter: [fonts, letters, pixels] -> mean cosine between each font's own letters."""
    unit = perLetter / (np.linalg.norm(perLetter, axis=2, keepdims=True) + 1e-8)
    iu = np.triu_indices(perLetter.shape[1], 1)
    return np.einsum("nid,njd->nij", unit, unit)[:, iu[0], iu[1]].mean(1)


def exactPairs(vectors, threshold, rowBlock=4096, colBlock=16384):
    """Every pair (i < j) of unit vectors with cosine > threshold, exact."""
    n = len(vectors)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else None
    except ImportError:
        device = None
    allI, allJ, allS = [], [], []
    started = time.time()
    for r in range(0, n, rowBlock):
        rows = vectors[r:r + rowBlock].astype(np.float32)
        if device:
            rows = torch.from_numpy(rows).to(device)
        for c in range(r, n, colBlock):
            cols = vectors[c:c + colBlock].astype(np.float32)
            if device:
                block = (rows @ torch.from_numpy(cols).to(device).T).cpu().numpy()
            else:
                block = rows @ cols.T
            ii, jj = np.nonzero(block > threshold)
            keep = (jj + c) > (ii + r)
            ii, jj = ii[keep], jj[keep]
            allI.append((ii + r).astype(np.int32)); allJ.append((jj + c).astype(np.int32))
            allS.append(block[ii, jj])
        done = min(r + rowBlock, n)
        work = 1 - (1 - done / n) ** 2
        print(f"\r  dedup pairs ({device or 'cpu'}): {done}/{n} rows, {sum(len(x) for x in allI)} pairs, "
              f"~{(time.time() - started) * (1 - work) / max(work, 1e-9) / 60:.0f} min left", end="", flush=True)
    print()
    if not allI:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    return np.concatenate(allI), np.concatenate(allJ), np.concatenate(allS)


def greedyDiscard(n, i, j, s, t, seed, preferred=None):
    """Boolean keep mask over n compared fonts: pairs with s > t in random order, drop one of each pair whose fonts are
    both still kept -- the non-preferred one if exactly one is preferred (bool mask over the n fonts), else at random."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.nonzero(s > t)[0])
    coins = (rng.random(len(order)) < 0.5).tolist()
    preferred = np.zeros(n, bool) if preferred is None else preferred
    alive = np.ones(n, bool)
    for a, b, coin in zip(i[order].tolist(), j[order].tolist(), coins):
        if alive[a] and alive[b]:
            if preferred[a] != preferred[b]:
                alive[b if preferred[a] else a] = False
            else:
                alive[a if coin else b] = False
    return alive


class NearDuplicates:
    """keys: one per font in the corpus (any hashable). compared: indices of fonts with every dedup letter;
    placeholder: bool over compared; (i, j, s): pairs above MIN_THRESHOLD, indexing into compared."""

    def __init__(self, keys, compared, placeholder, i, j, s, fingerprint=""):
        self.keys, self.compared, self.placeholder = list(keys), np.asarray(compared), np.asarray(placeholder)
        self.i, self.j, self.s = i, j, s
        self.fingerprint = fingerprint

    # ---------------------------------------------------------------- building

    @classmethod
    def build(cls, keys, glyphsOf, cachePath, fingerprint=""):
        """glyphsOf(index) -> list of the DEDUP_LETTERS glyph images for font `index`, or None if any is missing.
        Cached at cachePath (.npz); rebuilt unless both the corpus keys and the fingerprint match."""
        if os.path.exists(cachePath):
            cached = cls.load(cachePath)
            if cached.keys == list(keys) and cached.fingerprint == fingerprint:
                print(f"dedup: pairs cached in {cachePath}", flush=True)
                return cached
        started = time.time()
        keys = list(keys)
        vectorPath = cachePath + ".vectors.npy"
        os.makedirs(os.path.dirname(os.path.abspath(cachePath)), exist_ok=True)
        vectors = np.lib.format.open_memmap(vectorPath, mode="w+", dtype=np.float16,
                                            shape=(len(keys), len(DEDUP_LETTERS) * IMAGE_SIZE ** 2))
        compared, placeholder, real = [], [], []   # real: positions in `compared` that aren't placeholder fonts
        batch = []

        def flush():
            perLetter = np.stack(batch)
            isPlaceholder = interLetterSimilarity(perLetter) > PLACEHOLDER_SIMILARITY
            offset = len(placeholder)
            placeholder.extend(isPlaceholder.tolist())
            flat = perLetter[~isPlaceholder].reshape(int((~isPlaceholder).sum()), -1)
            flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
            vectors[len(real):len(real) + len(flat)] = flat
            real.extend((offset + np.nonzero(~isPlaceholder)[0]).tolist())
            batch.clear()

        for index in range(len(keys)):
            glyphs = glyphsOf(index)
            if glyphs is None:
                continue
            compared.append(index)
            batch.append(dedupVector(glyphs))
            if len(batch) == 2048:
                flush()
                print(f"\r  dedup vectors: {index + 1}/{len(keys)} fonts", end="", flush=True)
        if batch:
            flush()
        print(f"\r  dedup vectors: {len(compared)} of {len(keys)} fonts have all of {DEDUP_LETTERS} "
              f"({time.time() - started:.0f}s)", flush=True)

        placeholder = np.array(placeholder, bool)
        real = np.array(real, np.int64)
        i, j, s = exactPairs(vectors[:len(real)], MIN_THRESHOLD)
        i, j = real[i].astype(np.int32), real[j].astype(np.int32)
        result = cls(keys, compared, placeholder, i, j, s, fingerprint)
        result.save(cachePath)
        del vectors
        os.remove(vectorPath)
        print(f"dedup: {len(i)} pairs with s > {MIN_THRESHOLD}, {int(placeholder.sum())} placeholder-box fonts "
              f"({time.time() - started:.0f}s) -> {cachePath}", flush=True)
        return result

    @classmethod
    def fromPaths(cls, names, letters, paths, cacheDir=DEFAULT_CACHE_DIR):
        """Bitmap-cache format (loaders/standard.py, loaders/myfonts.py): parallel arrays of font name, letter (with
        its real case) and .bmp path."""
        pathMap = {}
        for name, letter, path in zip(names, letters, paths):
            if letter in DEDUP_LETTERS:
                pathMap.setdefault(name, {})[letter] = path
        keys = sorted(set(names.tolist() if hasattr(names, "tolist") else names))

        def glyphsOf(index):
            letterPaths = pathMap.get(keys[index], {})
            if any(letter not in letterPaths for letter in DEDUP_LETTERS):
                return None
            images = [cv2.imdecode(np.fromfile(letterPaths[letter], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                      for letter in DEDUP_LETTERS]
            if any(image is None for image in images):
                return None
            return [image.astype(np.float32) / 255.0 for image in images]

        digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]
        return cls.build(keys, glyphsOf, os.path.join(cacheDir, f"pairs_{digest}.npz"))

    @staticmethod
    def glyphCacheFingerprint(meta):
        return hashlib.sha256("\n".join(map(str, meta["names"])).encode("utf-8")).hexdigest()

    @staticmethod
    def glyphCacheMyFonts(meta):
        """Row indices of a glyph cache's MyFonts fonts (the preferred side of a near-duplicate pair)."""
        return {row for row, source in enumerate(meta["sources"]) if source == "myfonts"}

    @classmethod
    def loadGlyphCache(cls, dataDir):
        """The dedup.npz build_glyph_cache.py wrote, checked against the cache it belongs to."""
        with open(os.path.join(dataDir, "meta.json"), encoding="utf8") as f:
            meta = json.load(f)
        nearDuplicates = cls.load(os.path.join(dataDir, "dedup.npz"))
        if nearDuplicates.fingerprint != cls.glyphCacheFingerprint(meta):
            raise RuntimeError(f"{dataDir}/dedup.npz belongs to a different build of the cache; rebuild it")
        return nearDuplicates, cls.glyphCacheMyFonts(meta)

    @classmethod
    def fromGlyphCache(cls, dataDir):
        """experiments/levjepa/build_glyph_cache.py format; keys are row indices of glyphs.npy."""
        glyphs = np.load(os.path.join(dataDir, "glyphs.npy"), mmap_mode="r")
        present = np.load(os.path.join(dataDir, "present.npy"))
        with open(os.path.join(dataDir, "meta.json"), encoding="utf8") as f:
            meta = json.load(f)
        letterIndex = [meta["letters"].index(c) for c in DEDUP_LETTERS]
        comparable = present[:, letterIndex].all(1)

        def glyphsOf(index):
            if not comparable[index]:
                return None
            return [g.astype(np.float32) / 255.0 for g in glyphs[index, letterIndex]]

        return cls.build(list(range(len(glyphs))), glyphsOf, os.path.join(dataDir, "dedup.npz"),
                         cls.glyphCacheFingerprint(meta))

    def save(self, path):
        np.savez_compressed(path, keys=np.array(self.keys, dtype=object), compared=self.compared,
                            placeholder=self.placeholder, i=self.i, j=self.j, s=self.s,
                            fingerprint=np.array(self.fingerprint))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        return cls(data["keys"].tolist(), data["compared"], data["placeholder"], data["i"], data["j"], data["s"],
                   str(data["fingerprint"]) if "fingerprint" in data else "")

    # ---------------------------------------------------------------- using

    def preferredMask(self, preferred):
        """Bool mask over compared fonts for a collection of preferred keys (MyFonts)."""
        preferred = set(preferred or ())
        return np.array([self.keys[k] in preferred for k in self.compared], bool)

    def discarded(self, t, seed=0, preferred=None):
        """Keys dropped at threshold t: placeholder-box fonts plus the rule's discards (preferred keys win ties)."""
        if t < MIN_THRESHOLD:
            raise ValueError(f"pairs are only cached above {MIN_THRESHOLD}")
        alive = greedyDiscard(len(self.compared), self.i, self.j, self.s, t, seed,
                              self.preferredMask(preferred)) & ~self.placeholder
        return {self.keys[k] for k in self.compared[~alive]}

    def table(self, thresholds=THRESHOLDS, seeds=5, preferred=None):
        corpus = len(self.keys) - int(self.placeholder.sum())
        mask = self.preferredMask(preferred)
        rows = []
        for t in thresholds:
            kept = [int((greedyDiscard(len(self.compared), self.i, self.j, self.s, t, seed, mask)
                         & ~self.placeholder).sum())
                    for seed in range(seeds)]
            discarded = len(self.compared) - int(self.placeholder.sum()) - float(np.mean(kept))
            rows.append({"t": t, "pairs": int((self.s > t).sum()), "discarded": discarded,
                         "percent": 100 * discarded / corpus, "after": corpus - discarded})
        return {"fonts": len(self.keys), "notCompared": len(self.keys) - len(self.compared),
                "placeholder": int(self.placeholder.sum()), "corpus": corpus, "rows": rows}

    def printTable(self, thresholds=THRESHOLDS, seeds=5, preferred=None):
        table = self.table(thresholds, seeds, preferred)
        print(f"\nfonts {table['fonts']}: {table['placeholder']} placeholder-box fonts dropped, "
              f"{table['notCompared']} missing a letter of {DEDUP_LETTERS} (kept, not compared)")
        print(f"corpus before dedup: {table['corpus']}\n")
        print("| t | pairs with s > t | discarded | % discarded | corpus after |")
        print("|---|---|---|---|---|")
        print(f"| none | | 0 | 0.0% | {table['corpus']} |")
        for row in table["rows"]:
            print(f"| {row['t']} | {row['pairs']} | {row['discarded']:.0f} | {row['percent']:.1f}% | "
                  f"{row['after']:.0f} |")
        return table


def applyDedup(config, names, letters, paths, preferred=()):
    """Hook for datasets that concatenate several sources' (names, letters, paths). Always drops repeated
    (font, letter) entries (the same font file in two directories has the same hashed key); with the config key
    "dedupThreshold" also drops placeholder-box fonts and near-duplicates (NearDuplicates.discarded, seed 0),
    keeping fonts in `preferred` (the MyFonts names) over the rest."""
    seen, keep = set(), np.zeros(len(names), bool)
    for index, entry in enumerate(zip(names.tolist(), letters.tolist())):
        if entry not in seen:
            seen.add(entry)
            keep[index] = True
    if (~keep).any():
        print(f"dedup: {(~keep).sum()} glyphs repeated across sources (same font file) dropped")
    names, letters, paths = names[keep], letters[keep], paths[keep]

    if "dedupThreshold" in config:
        cacheDir = config.dedupCacheDir if "dedupCacheDir" in config else DEFAULT_CACHE_DIR
        nearDuplicates = NearDuplicates.fromPaths(names, letters, paths, cacheDir)
        dropped = nearDuplicates.discarded(config.dedupThreshold, preferred=preferred)
        keep = ~np.isin(names, list(dropped))
        print(f"dedup: t={config.dedupThreshold} drops {len(dropped)} of {len(nearDuplicates.keys)} fonts "
              f"({(~keep).sum()} glyphs)")
        names, letters, paths = names[keep], letters[keep], paths[keep]
    return names, letters, paths


def main():
    p = argparse.ArgumentParser(description="thresholds vs. fonts discarded over the compiled corpus")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="config JSON with a dataset.directories block (bitmap caches)")
    source.add_argument("--glyphCache", help="experiments/levjepa/build_glyph_cache.py output directory")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    if args.glyphCache:
        nearDuplicates = NearDuplicates.fromGlyphCache(args.glyphCache)
        with open(os.path.join(args.glyphCache, "meta.json"), encoding="utf8") as f:
            preferred = NearDuplicates.glyphCacheMyFonts(json.load(f))
    else:
        from utils.config import Config
        from utils.loaders.myfonts import loadMyFontsImagePaths
        from utils.loaders.standard import collectFontSetPaths
        dataset = Config().load(args.config).dataset
        cacheSuffix = dataset.cacheSuffix if "cacheSuffix" in dataset else ""
        parts, preferred = [], set()
        if "myFonts" in dataset.directories:
            parts.append(loadMyFontsImagePaths(dataset.directories.myFonts, dataset.fontSize, cacheSuffix=cacheSuffix))
            preferred = set(parts[-1]["names"].tolist())
        for directory in (dataset.directories.standard if "standard" in dataset.directories else []):
            parts.append(collectFontSetPaths(directory, dataset.fontSize, dataset.maps, cacheSuffix=cacheSuffix))
        names, letters, paths = (np.concatenate([part[k] for part in parts]) for k in ("names", "letters", "paths"))
        names, letters, paths = applyDedup({}, names, letters, paths)
        nearDuplicates = NearDuplicates.fromPaths(names, letters, paths,
                                                  dataset.dedupCacheDir if "dedupCacheDir" in dataset
                                                  else DEFAULT_CACHE_DIR)
    nearDuplicates.printTable(seeds=args.seeds, preferred=preferred)


if __name__ == "__main__":
    main()
