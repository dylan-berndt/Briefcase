# Tests the premise behind a proposed fix: weight each descriptor (tag or
# adjective) in the vocabulary by how visually coherent it actually is,
# instead of treating every descriptor as equally trustworthy input to a
# captioner. If some descriptors show strong coherence (fonts sharing that
# descriptor really do cluster together in embeddings/all.json) while
# others don't, that's real support for the weighting idea. If almost none
# show real coherence, the problem isn't which descriptors to trust more --
# the raw tag/adjective vocabulary itself carries little visual signal, and
# no downstream weighting scheme built only from these tags can fix that.
#
# This also indirectly answers "did the embedding->vocabulary mapping
# attempt have a bug, or was there nothing to learn": if per-descriptor
# coherence is near zero almost everywhere, a learned classifier predicting
# descriptors from embeddings has nothing real to fit either -- that's not
# an implementation bug, it's an absence of signal.
#
# Method: for each descriptor, take the set of fonts carrying it, compute
# their mean pairwise cosine similarity in the font-embedding space, and
# compare against a random-sample-of-the-same-size baseline (needed because
# the embedding space isn't perfectly isotropic, so raw within-group
# similarity alone isn't interpretable without a null to compare to).

import json
import random

import numpy as np
from pygtrie import CharTrie

from utils import Config, loadDescriptionsFromSource

EMBEDDINGS_PATH = "embeddings/all.json"
OUTPUT_PATH = "results/descriptorGrounding.json"
MIN_FONTS_PER_DESCRIPTOR = 30
BASELINE_SAMPLES = 20
SEED = 0


def loadFontEmbeddings(path=EMBEDDINGS_PATH):
    with open(path) as f:
        data = json.load(f)
    return {name: np.array(v, dtype=np.float32) for name, v in data.items()}


def matchToEmbeddings(fontEmbeddings, descriptions):
    trie = CharTrie()
    for key in descriptions:
        trie[key] = key
    candidates = {}
    for embName, vector in fontEmbeddings.items():
        match = trie.longest_prefix(embName.strip())
        if match.key is None:
            continue
        candidates.setdefault(match.key, []).append((embName, vector))
    return {k: min(v, key=lambda p: len(p[0]))[1] for k, v in candidates.items()}


def meanPairwiseSim(vectors):
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sim = v @ v.T
    iu = np.triu_indices(len(v), k=1)
    return float(sim[iu].mean())


def main():
    rng = random.Random(SEED)

    print("Loading font embeddings ...")
    fontEmbeddings = loadFontEmbeddings()

    print("Loading raw tag/adjective descriptions (pulls in spacy/cv2/transformers, ~1 min) ...")
    config = Config().load("configs/vit.json")
    descriptions = loadDescriptionsFromSource(config.dataset)

    matched = matchToEmbeddings(fontEmbeddings, descriptions)
    print(f"{len(matched)} fonts matched to embeddings")

    descriptorFonts = {}
    for name in matched:
        desc = descriptions[name]
        for tag, weight in desc.tags.items():
            if weight > 0.3:
                descriptorFonts.setdefault(tag, set()).add(name)
        for adjective in desc.adjectives:
            descriptorFonts.setdefault(adjective, set()).add(name)

    allNames = list(matched.keys())
    allVectors = np.stack([matched[n] for n in allNames])
    print(f"{len(descriptorFonts)} distinct descriptors total")

    results = []
    for descriptor, fontSet in descriptorFonts.items():
        fonts = list(fontSet)
        if len(fonts) < MIN_FONTS_PER_DESCRIPTOR:
            continue
        vectors = np.stack([matched[n] for n in fonts])
        withinSim = meanPairwiseSim(vectors)

        baselineSims = [
            meanPairwiseSim(allVectors[rng.sample(range(len(allNames)), len(fonts))])
            for _ in range(BASELINE_SAMPLES)
        ]
        baselineMean = float(np.mean(baselineSims))
        baselineStd = float(np.std(baselineSims))
        score = withinSim - baselineMean
        # z-score against the descriptor's OWN baseline noise, not a flat
        # score cutoff -- a small-n descriptor's baseline estimate is
        # noisier, so it needs a bigger raw margin to mean the same thing
        # as a large-n descriptor's smaller margin. This is what actually
        # answers "where's the noise floor," not an arbitrary round number.
        z = score / max(baselineStd, 1e-6)
        results.append({
            "descriptor": descriptor, "n": len(fonts), "within": withinSim,
            "baselineMean": baselineMean, "baselineStd": baselineStd,
            "score": score, "z": z,
        })

    results.sort(key=lambda r: r["z"], reverse=True)

    print(f"\n{len(results)} descriptors with >= {MIN_FONTS_PER_DESCRIPTOR} fonts\n")
    print("=== TOP 25 most visually coherent descriptors (by z-score) ===")
    for r in results[:25]:
        print(f"  {r['descriptor']:<25} n={r['n']:<6} within={r['within']:.4f}  "
              f"baseline={r['baselineMean']:.4f}±{r['baselineStd']:.4f}  score={r['score']:+.4f}  z={r['z']:+.1f}")

    print("\n=== BOTTOM 25 least coherent descriptors (by z-score) ===")
    for r in results[-25:]:
        print(f"  {r['descriptor']:<25} n={r['n']:<6} within={r['within']:.4f}  "
              f"baseline={r['baselineMean']:.4f}±{r['baselineStd']:.4f}  score={r['score']:+.4f}  z={r['z']:+.1f}")

    zScores = np.array([r["z"] for r in results])
    print(f"\n=== overall distribution ({len(results)} descriptors) ===")
    print(f"  z mean={zScores.mean():.2f}  median={np.median(zScores):.2f}")
    for zCut in (1.0, 1.65, 2.0, 3.0):
        frac = (zScores > zCut).mean() * 100
        print(f"  fraction with z > {zCut:<4}: {frac:5.1f}%")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full per-descriptor results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
