"""
Same ASIF method/hyperparameters as asif_retrieval.py's font run, pointed at an arbitrary
embeddings/<name>.json instead of the hardcoded all.json, so new checkpoints can be measured
directly instead of by geometry proxy.

    python experiments/embedding-geometry/asif_retrieval_checkpoint.py <embeddingsName> [maxFonts]
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asif_retrieval import evaluateRetrieval, K_VALUES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def loadFonts(embeddingsName, maxFonts=None, abttRemove=0):
    with open(os.path.join(REPO_ROOT, "embeddings", f"{embeddingsName}.json")) as f:
        fontEmbeddings = json.load(f)
    with open(os.path.join(REPO_ROOT, "embeddings", "sentenceQueries_BAAI_bge-large-en-v1.5.pkl"), "rb") as f:
        sentenceCache = pickle.load(f)
    names = sorted(set(fontEmbeddings.keys()) & set(sentenceCache.keys()))
    allImageVecs = np.array([fontEmbeddings[n] for n in names], dtype=np.float64)
    if abttRemove > 0:
        from abtt_postprocess import allButTheTop
        allImageVecs = allButTheTop(allImageVecs, abttRemove)
    imageByName = dict(zip(names, allImageVecs))
    if maxFonts is not None:
        names = names[:maxFonts]
    imageVecs = np.array([imageByName[n] for n in names], dtype=np.float64)
    textVecs, textToImageIdx = [], []
    for i, n in enumerate(names):
        for v in sentenceCache[n]:
            textVecs.append(v)
            textToImageIdx.append(i)
    return imageVecs, np.stack(textVecs).astype(np.float64), np.array(textToImageIdx)


def main():
    embeddingsName = sys.argv[1]
    maxFonts = int(sys.argv[2]) if len(sys.argv) > 2 else 8091
    abttRemove = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    imageVecs, textVecs, textToImageIdx = loadFonts(embeddingsName, maxFonts=maxFonts, abttRemove=abttRemove)
    print(f"{embeddingsName}: {imageVecs.shape[0]} fonts (dim {imageVecs.shape[1]}), {textVecs.shape[0]} queries", flush=True)
    numAnchors = min(6000, imageVecs.shape[0] - 500)
    numQueries = min(500, imageVecs.shape[0] - numAnchors)
    recall, n, medianRank, meanRank = evaluateRetrieval(imageVecs, textVecs, textToImageIdx, numAnchors, numQueries)
    print(f"anchors={numAnchors} query queries={n} (from {numQueries} held-out fonts)")
    for k in K_VALUES:
        print(f"  recall@{k}: {recall[k]*100:.2f}%")
    print(f"  median rank: {medianRank:.0f} / {imageVecs.shape[0]}  mean rank: {meanRank:.0f}")


if __name__ == "__main__":
    main()
