"""
Re-runs the font-side ASIF retrieval eval (same protocol as asif_retrieval.py:
matched to Flickr8k's 8,091-item scale, 6,000 anchors) with All-but-the-Top
applied to the visual embeddings first, to check whether the geometric
improvement abtt_postprocess.py measured (participation ratio 57.6->71.3,
NN-cosine 0.962->0.884 at numRemove=20) actually translates into better
retrieval, not just better diagnostic numbers.

    python3 experiments/embedding-geometry/asif_retrieval_abtt.py
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
if _scriptDir not in _sys.path:
    _sys.path.insert(0, _scriptDir)

from asif_retrieval import loadFonts, evaluateRetrieval, K_VALUES


def runFontEval(abttRemove):
    imageVecs, textVecs, textToImageIdx = loadFonts(maxFonts=8091, abttRemove=abttRemove)
    numAnchors = min(6000, imageVecs.shape[0] - 500)
    numQueries = min(500, imageVecs.shape[0] - numAnchors)
    recall, n, medianRank, meanRank = evaluateRetrieval(imageVecs, textVecs, textToImageIdx,
                                                          numAnchors, numQueries)
    print(f"\n=== FONTS, ABTT remove-top-{abttRemove} ===")
    print(f"anchors={numAnchors} query queries={n} (from {numQueries} held-out fonts)")
    for k in K_VALUES:
        print(f"  recall@{k}: {recall[k]:.4f}")
    print(f"  median rank: {medianRank:.0f} / {imageVecs.shape[0]}  mean rank: {meanRank:.0f}")


def main():
    for abttRemove in [0, 10, 20, 50]:
        runFontEval(abttRemove)


if __name__ == "__main__":
    main()
