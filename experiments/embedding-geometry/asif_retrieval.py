"""
ASIF-style (Norelli et al., NeurIPS 2023, arxiv.org/abs/2210.01738) training-
free cross-modal retrieval: given two INDEPENDENTLY pretrained, never-
jointly-trained encoders (an image/visual encoder and a text encoder) plus
a modest number of paired examples, align them with no training at all.

Method: pick N anchor pairs. A new item's "relative representation" is the
vector of its cosine similarities to the N anchor points IN ITS OWN
MODALITY'S embedding space (images compared to anchor images, text compared
to anchor captions). Because the anchors are paired, both relative
representations end up indexed by the same N anchor ids -- directly
comparable even though they came from unrelated encoders/spaces. Sparsify
(zero all but the top-k similarities) and exponentiate (raise to power p)
to sharpen the signal, then rank candidates by cosine similarity between
relative representations.

Runs the SAME method on both Flickr8k (DINOv2 + BGE, the known-working
reference) and this project's own fonts (project ViT + BGE), so any
difference in recall@k is attributable to the embeddings' own geometry,
not to a different retrieval algorithm.

    python3 experiments/embedding-geometry/asif_retrieval.py
"""
import json
import os
import pickle

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

K_VALUES = [1, 5, 10, 50, 100]


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def relativeRepresentations(queryVecs, anchorVecs, topK=800, power=8):
    """queryVecs [M, d], anchorVecs [N, d] (SAME modality/space for both) -> [M, N] sparsified,
    exponentiated similarity matrix, matching ASIF's own hyperparameters (topK=800, power=8)."""
    sims = normalize(queryVecs) @ normalize(anchorVecs).T  # [M, N]
    if topK < sims.shape[1]:
        threshold = np.partition(sims, -topK, axis=1)[:, -topK][:, None]
        sims = np.where(sims >= threshold, sims, 0.0)
    sims = np.sign(sims) * np.abs(sims) ** power
    return sims


def evaluateRetrieval(imageVecsAll, textVecsAll, textToImageIdx, numAnchors, numQueries, seed=0, topK=800, power=8):
    """
    imageVecsAll: [numItems, dImg] one embedding per item (e.g. per image/font).
    textVecsAll: [numTexts, dTxt] one embedding per text (e.g. per caption/query) --
        possibly several per item.
    textToImageIdx: [numTexts] which item (row of imageVecsAll) each text belongs to.
    Splits items into anchors (used to build the relative-representation basis) and a held-out
    query set (their TEXT is used to retrieve among ALL items' images, matching this project's
    own "text query -> which of the N items" retrieval framing). Anchors' own texts are excluded
    from the query set but their IMAGES remain valid retrieval targets (an anchor's image can
    still be a correct answer for someone else's held-out caption if they're duplicates, though
    with distinct items this shouldn't matter here).
    """
    rng = np.random.RandomState(seed)
    numItems = imageVecsAll.shape[0]
    order = rng.permutation(numItems)
    anchorItems = set(order[:numAnchors].tolist())
    queryItems = order[numAnchors:numAnchors + numQueries]

    anchorImageVecs = imageVecsAll[list(anchorItems)]
    anchorItemsList = list(anchorItems)
    # anchor captions: one per anchor item (first caption) to keep anchors/pairs 1:1, matching
    # ASIF's own "N paired examples" framing
    anchorTextIdx = []
    itemToFirstText = {}
    for ti, item in enumerate(textToImageIdx):
        if item not in itemToFirstText:
            itemToFirstText[item] = ti
    for item in anchorItemsList:
        anchorTextIdx.append(itemToFirstText[item])
    anchorTextVecs = textVecsAll[anchorTextIdx]

    # image relative representations for ALL items (retrieval candidates)
    imageRel = relativeRepresentations(imageVecsAll, anchorImageVecs, topK=min(topK, len(anchorItemsList)), power=power)

    # query texts: all captions belonging to queryItems, excluding anchor items' own captions
    queryItemSet = set(queryItems.tolist())
    queryTextIdx = [ti for ti, item in enumerate(textToImageIdx) if item in queryItemSet]
    queryTextVecs = textVecsAll[queryTextIdx]
    queryTrueItem = [textToImageIdx[ti] for ti in queryTextIdx]

    textRel = relativeRepresentations(queryTextVecs, anchorTextVecs, topK=min(topK, len(anchorItemsList)), power=power)

    simMatrix = normalize(textRel) @ normalize(imageRel).T  # [numQueryTexts, numItems]

    hits = {k: 0 for k in K_VALUES}
    ranks = []
    for qi in range(simMatrix.shape[0]):
        order = np.argsort(-simMatrix[qi])
        trueItem = queryTrueItem[qi]
        rank = int(np.where(order == trueItem)[0][0]) + 1
        ranks.append(rank)
        for k in K_VALUES:
            if rank <= k:
                hits[k] += 1

    n = simMatrix.shape[0]
    recall = {k: hits[k] / n for k in K_VALUES}
    return recall, n, np.median(ranks), np.mean(ranks)


def loadFlickr8k():
    with open(os.path.join(SCRIPT_DIR, "data", "image_embeddings.pkl"), "rb") as f:
        images = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, "data", "text_embeddings.pkl"), "rb") as f:
        texts = pickle.load(f)
    names = sorted(set(images.keys()) & set(texts.keys()))
    imageVecs = np.stack([images[n] for n in names])
    textVecs, textToImageIdx = [], []
    for i, n in enumerate(names):
        for v in texts[n]:
            textVecs.append(v)
            textToImageIdx.append(i)
    return imageVecs, np.stack(textVecs), np.array(textToImageIdx)


def loadFonts(maxFonts=None, abttRemove=0):
    with open(os.path.join(REPO_ROOT, "embeddings", "all.json")) as f:
        fontEmbeddings = json.load(f)
    with open(os.path.join(REPO_ROOT, "embeddings", "sentenceQueries_BAAI_bge-large-en-v1.5.pkl"), "rb") as f:
        sentenceCache = pickle.load(f)
    names = sorted(set(fontEmbeddings.keys()) & set(sentenceCache.keys()))

    # ABTT's top-D directions are a property of the FULL corpus's covariance -- compute them
    # over all matched fonts (not just the maxFonts subset used for the retrieval eval) so this
    # matches exactly what abtt_postprocess.py's diagnostic measured.
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
    print("=== FLICKR8K (DINOv2 images / BGE captions, known-working reference) ===")
    imageVecs, textVecs, textToImageIdx = loadFlickr8k()
    print(f"{imageVecs.shape[0]} images, {textVecs.shape[0]} captions")
    numAnchors = min(6000, imageVecs.shape[0] - 500)
    numQueries = min(500, imageVecs.shape[0] - numAnchors)
    recall, n, medianRank, meanRank = evaluateRetrieval(imageVecs, textVecs, textToImageIdx,
                                                          numAnchors, numQueries)
    print(f"anchors={numAnchors} query captions={n} (from {numQueries} held-out images)")
    for k in K_VALUES:
        print(f"  recall@{k}: {recall[k]:.4f}")
    print(f"  median rank: {medianRank:.0f} / {imageVecs.shape[0]}  mean rank: {meanRank:.0f}")

    print("\n=== FONTS (project ViT visual / BGE queries) ===")
    imageVecs, textVecs, textToImageIdx = loadFonts(maxFonts=8091)  # match Flickr8k's item count
    print(f"{imageVecs.shape[0]} fonts, {textVecs.shape[0]} queries")
    numAnchors = min(6000, imageVecs.shape[0] - 500)
    numQueries = min(500, imageVecs.shape[0] - numAnchors)
    recall, n, medianRank, meanRank = evaluateRetrieval(imageVecs, textVecs, textToImageIdx,
                                                          numAnchors, numQueries)
    print(f"anchors={numAnchors} query queries={n} (from {numQueries} held-out fonts)")
    for k in K_VALUES:
        print(f"  recall@{k}: {recall[k]:.4f}")
    print(f"  median rank: {medianRank:.0f} / {imageVecs.shape[0]}  mean rank: {meanRank:.0f}")


if __name__ == "__main__":
    main()
