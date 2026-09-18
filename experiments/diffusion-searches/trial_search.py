"""
Interactive human trial for the classifier-driven branching search
(v4 config: checkpoints/nav_classifier_v4, hard-commit, no beam --
beam's hedge branches aren't something a user directly picks, so this
tests the base decision loop the whole investigation's numbers rest
on). A random held-out (font, query) pair is picked as the session's
target; its glyphs are shown in a fixed corner box for the whole
session, alongside the real text query that's driving the classifier.
At each of up to 5 rounds, the classifier's top-3 candidate children
are shown as three scrollable columns of ~24 real member-font
previews each (sampled by distance-to-child-centroid, most
representative first) -- press 1/2/3 to pick the column you think
contains fonts closest to the target.

Glyphs are rendered directly from each font's own .ttf/.otf file via
pygame's native font rendering (build_font_path_cache.py's name->path
lookup) -- no PIL bitmap preprocessing pipeline, which would eagerly
render the entire ~39k-font corpus just to answer "what does this one
font look like."

No correctness feedback is shown mid-session (a real user wouldn't
get that either) -- every decision is logged to trial_log.jsonl
(append-only, one line per session) with whether it matched the
TRUE tree path, so accumulated sessions give an empirical human
error/noise rate directly comparable to the noiseProb values (0.1,
0.2) used throughout this investigation's synthetic testing.

Controls: [1]/[2]/[3] pick a column. [Mouse wheel] scroll. [N] start
a new session (fresh random target) once the current one ends.
[Esc]/close window to quit.

    python3 experiments/diffusion-searches/trial_search.py
"""
import argparse
import datetime
import json
import os
import pickle
import sys

# First experiments/diffusion-searches script needing repo-root packages
# (build_font_path_cache lives alongside it, but nothing outside this dir).
# Running as `python3 experiments/diffusion-searches/trial_search.py` puts
# the script's own directory on sys.path, not the repo root -- add both
# explicitly so relative data paths (checkpoints/, embeddings/, etc.)
# resolve regardless of the invoking working directory.
_scriptDir = os.path.dirname(os.path.abspath(__file__))
_repoRoot = os.path.dirname(os.path.dirname(_scriptDir))
for _p in (_repoRoot, _scriptDir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pygame
import torch

from build_font_path_cache import loadOrBuildCache
from corpus import FontCorpus, HierarchicalClusterIndex
from dataset import PCAWhitener, EMBEDDINGS_PATH, loadDescriptions
from train_navigation_classifier import NavigationClassifier, trueChildAt

PREVIEW_TEXT = "AaBbGg"
LOG_PATH = "experiments/diffusion-searches/trial_log.jsonl"


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierDir", default="checkpoints/nav_classifier_v4")
    parser.add_argument("--maxOptions", type=int, default=3)
    parser.add_argument("--maxRounds", type=int, default=5)
    parser.add_argument("--maxDepth", type=int, default=5)
    parser.add_argument("--acceptanceSize", type=int, default=10)
    parser.add_argument("--previewsPerColumn", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None, help="Fix the random target/query for reproducibility.")
    return parser.parse_args()


def cachePathFor(modelName):
    return os.path.join("embeddings", f"sentenceQueries_{modelName.replace('/', '_')}.pkl")


def scoreChildren(model, text, node, device):
    childCentroids = np.stack([c.centroid for c in node.children]).astype(np.float32)
    numChildren = childCentroids.shape[0]
    textT = text.unsqueeze(0)
    nodeT = torch.from_numpy(node.centroid.astype(np.float32)).unsqueeze(0).to(device)
    childrenT = torch.from_numpy(childCentroids).unsqueeze(0).to(device)
    maskT = torch.ones(1, numChildren, dtype=torch.bool, device=device)
    with torch.no_grad():
        scores = model(textT, nodeT, childrenT, maskT).squeeze(0)
    return torch.softmax(scores, dim=0).cpu().numpy()


def representativeMembers(node, corpus, limit):
    members = node.memberIndices
    if len(members) == 0:
        return []
    memberVecs = corpus.whitenedMatrix[members]
    centroid = torch.from_numpy(node.centroid.astype(np.float32)).to(corpus.device)
    dist = torch.norm(memberVecs - centroid.unsqueeze(0), dim=1)
    order = dist.argsort().cpu().numpy()
    return [corpus.names[members[i]] for i in order[:limit]]


def rankInLeaf(node, corpus, targetIdx):
    members = node.memberIndices
    if len(members) == 0:
        return None, 0
    memberVecs = corpus.whitenedMatrix[members]
    centroid = torch.from_numpy(node.centroid.astype(np.float32)).to(corpus.device)
    dist = torch.norm(memberVecs - centroid.unsqueeze(0), dim=1)
    order = dist.argsort().cpu().numpy()
    orderedMembers = [members[i] for i in order]
    rank = orderedMembers.index(targetIdx) + 1 if targetIdx in orderedMembers else None
    return rank, len(members)


class Session:
    def __init__(self, args, corpus, hier, model, sentenceCache, descriptions, testPairs, device,
                  fontPathMap, rng):
        self.args = args
        self.corpus = corpus
        self.hier = hier
        self.model = model
        self.device = device
        self.fontPathMap = fontPathMap

        candidates = [p for p in testPairs if p["font"] in corpus.nameToIndex and p["font"] in fontPathMap
                       and p["font"] in descriptions and p["index"] < len(descriptions[p["font"]])]
        pair = candidates[rng.randint(len(candidates))]
        self.targetName = pair["font"]
        self.targetIdx = corpus.nameToIndex[self.targetName]
        self.textTensor = torch.from_numpy(np.asarray(sentenceCache[pair["font"]][pair["index"]],
                                                          dtype=np.float32))
        self.queryDisplay = descriptions[pair["font"]][pair["index"]]

        self.node = hier.root
        self.depth = 0
        self.rounds = 0
        self.finished = False
        self.decisions = []  # log entries
        self.candidateGroups = []  # current round's 3 columns of font names
        self.candidateNodeIdx = []  # which child index each column corresponds to
        self._advanceOrPrepareRound()

    def _prepareRound(self):
        probs = scoreChildren(self.model, self.textTensor, self.node, self.device)
        k = min(self.args.maxOptions, len(self.node.children))
        candidateIdx = np.argsort(-probs)[:k].tolist()
        self.candidateNodeIdx = candidateIdx
        self.candidateGroups = [representativeMembers(self.node.children[c], self.corpus,
                                                         self.args.previewsPerColumn) for c in candidateIdx]

    def _advanceOrPrepareRound(self):
        if not self.node.children or self.depth >= self.args.maxDepth:
            self.finished = True
            return
        self._prepareRound()

    def choose(self, columnIdx):
        if self.finished or columnIdx >= len(self.candidateNodeIdx):
            return
        chosenChildIdx = self.candidateNodeIdx[columnIdx]
        trueIdx, _ = trueChildAt(self.node, self.targetIdx)
        wasCorrect = (trueIdx == chosenChildIdx)
        self.decisions.append({
            "round": self.rounds, "depth": self.depth,
            "shownChildren": self.candidateNodeIdx, "chosenChildIdx": chosenChildIdx,
            "trueChildIdx": trueIdx, "wasCorrect": bool(wasCorrect),
        })
        self.node = self.node.children[chosenChildIdx]
        self.depth += 1
        self.rounds += 1
        if self.rounds >= self.args.maxRounds:
            self.finished = True
            return
        self._advanceOrPrepareRound()

    def finalSummary(self):
        rank, leafSize = rankInLeaf(self.node, self.corpus, self.targetIdx)
        leafSuccess = rank is not None
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "targetFont": self.targetName,
            "rounds": self.rounds,
            "decisions": self.decisions,
            "leafSuccess": leafSuccess,
            "rankInLeaf": rank,
            "leafSize": leafSize,
        }


def main():
    args = parseArgs()
    device = "cpu"
    rng = np.random.RandomState(args.seed)

    print("Loading font file path cache (builds it on first run, then reused instantly)...")
    fontPathMap = loadOrBuildCache()

    with open(os.path.join(args.classifierDir, "config.json")) as f:
        clsConfig = json.load(f)
    whitener = PCAWhitener.load(os.path.join(clsConfig["baseCheckpoint"], "whitener.npz"))
    corpus = FontCorpus.load(whitener, EMBEDDINGS_PATH, device=device)
    hier = HierarchicalClusterIndex.load(clsConfig["treeCache"])

    with open(cachePathFor(clsConfig["sentenceModel"]), "rb") as f:
        sentenceCache = pickle.load(f)
    with open(os.path.join(clsConfig["baseCheckpoint"], "test_pairs.json")) as f:
        testPairs = json.load(f)
    descriptions = loadDescriptions()

    model = NavigationClassifier(clsConfig["textDim"], clsConfig["pcaDim"], clsConfig["hiddenDim"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.classifierDir, "checkpoint.pt"), map_location=device))
    model.eval()

    pygame.init()
    windowSize = [1500, 900]
    window = pygame.display.set_mode(windowSize)
    clock = pygame.time.Clock()

    uiFont = pygame.font.SysFont("Calibri", 22, bold=True)
    smallFont = pygame.font.SysFont("Calibri", 15)
    tinyFont = pygame.font.SysFont("Calibri", 13)

    headerCorner = (30, 16)
    targetCorner = (windowSize[0] - 340, 16)
    optionsCorner = (30, 130)
    COLUMNS = args.maxOptions
    COLUMN_WIDTH = (windowSize[0] - optionsCorner[0] - 20) // COLUMNS

    fontObjects = {}
    renderCache = {}

    def previewSurface(name, size, color=(255, 255, 255)):
        cacheKey = (name, size, color)
        if cacheKey in renderCache:
            return renderCache[cacheKey]
        path = fontPathMap.get(name)
        surface = None
        if path is not None:
            fontKey = (path, size)
            font = fontObjects.get(fontKey)
            if font is None:
                try:
                    font = pygame.font.Font(path, size)
                except Exception:
                    font = False  # cache the failure too, don't retry every frame
                fontObjects[fontKey] = font
            if font:
                try:
                    surface = font.render(PREVIEW_TEXT, True, color)
                except Exception:
                    surface = None
        renderCache[cacheKey] = surface
        return surface

    def logSession(summary):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(summary) + "\n")

    session = Session(args, corpus, hier, model, sentenceCache, descriptions, testPairs, device, fontPathMap, rng)
    scrollOffset = 0
    running = True

    while running:
        window.fill((28, 28, 40))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n and session.finished:
                    session = Session(args, corpus, hier, model, sentenceCache, descriptions, testPairs,
                                        device, fontPathMap, rng)
                    scrollOffset = 0
                elif not session.finished and pygame.K_1 <= event.key <= pygame.K_9:
                    col = event.key - pygame.K_1
                    if col < COLUMNS:
                        session.choose(col)
                        scrollOffset = 0
                        if session.finished:
                            logSession(session.finalSummary())
            elif event.type == pygame.MOUSEWHEEL:
                scrollOffset = max(0, scrollOffset - event.y * 24)

        # target box, top-right, always visible
        pygame.draw.rect(window, (50, 50, 70), (*targetCorner, 310, 100))
        window.blit(smallFont.render("TARGET:", False, [255, 220, 130]), (targetCorner[0] + 8, targetCorner[1] + 4))
        window.blit(smallFont.render(session.targetName[:34], False, [255, 255, 255]),
                    (targetCorner[0] + 8, targetCorner[1] + 24))
        targetImg = previewSurface(session.targetName, 44, color=(255, 255, 255))
        if targetImg is not None:
            window.blit(targetImg, (targetCorner[0] + 8, targetCorner[1] + 46))

        header = f"Query: \"{session.queryDisplay}\"   |   round {session.rounds + (0 if session.finished else 1)}" \
                 f"/{args.maxRounds}   |   depth {session.depth}"
        window.blit(uiFont.render(header, False, [255, 255, 255]), headerCorner)

        contentSurface = pygame.Surface(windowSize, pygame.SRCALPHA)

        if session.finished:
            summary = session.finalSummary()
            foundWord = "FOUND" if summary["leafSuccess"] else "NOT in"
            rankPart = f", rank {summary['rankInLeaf']}" if summary["rankInLeaf"] else ""
            resultText = (f"SESSION DONE -- target {foundWord} final leaf (size {summary['leafSize']}"
                          f"{rankPart}). [N] new session  [Esc] quit")
            window.blit(smallFont.render(resultText, False, [180, 255, 180]), (optionsCorner[0], 96))

            members = session.node.memberIndices.tolist()
            centroid = torch.from_numpy(session.node.centroid.astype(np.float32)).to(corpus.device)
            memberVecs = corpus.whitenedMatrix[members]
            order = torch.norm(memberVecs - centroid.unsqueeze(0), dim=1).argsort().cpu().numpy()
            orderedNames = [corpus.names[members[i]] for i in order]

            y = optionsCorner[1] - scrollOffset
            perRow = 6
            colW = (windowSize[0] - optionsCorner[0]) // perRow
            for i, name in enumerate(orderedNames):
                col = i % perRow
                row = i // perRow
                x = optionsCorner[0] + col * colW
                rowY = y + row * 90
                isTarget = (name == session.targetName)
                nameColor = (255, 230, 100) if isTarget else (255, 255, 255)
                contentSurface.blit(tinyFont.render(name[:22], False, nameColor), (x, rowY))
                img = previewSurface(name, 26, color=nameColor)
                if img is not None:
                    contentSurface.blit(img, (x, rowY + 16))
            totalHeight = (len(orderedNames) // perRow + 1) * 90
        else:
            window.blit(smallFont.render("Which column looks closest to the TARGET font shown top-right? "
                                          "Press 1/2/3.", False, [200, 200, 255]), (optionsCorner[0], 96))
            columnHeights = [-scrollOffset] * COLUMNS
            for col, names in enumerate(session.candidateGroups):
                x = optionsCorner[0] + col * COLUMN_WIDTH
                y = columnHeights[col]
                label = uiFont.render(f"[{col + 1}]  ({len(names)} shown)", False, [255, 220, 130])
                contentSurface.blit(label, (x, y))
                itemHeight = label.get_height() + 8
                for name in names:
                    contentSurface.blit(tinyFont.render(name[:26], False, [255, 255, 255]), (x, y + itemHeight))
                    itemHeight += 14
                    img = previewSurface(name, 24)
                    if img is not None:
                        contentSurface.blit(img, (x, y + itemHeight))
                        itemHeight += img.get_height() + 8
                columnHeights[col] = y + itemHeight + 20
            totalHeight = max(h + scrollOffset for h in columnHeights) if columnHeights else 0

        scrollOffset = min(scrollOffset, max(0, totalHeight - (windowSize[1] - optionsCorner[1])))
        window.blit(contentSurface, (0, optionsCorner[1]))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
