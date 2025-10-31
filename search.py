from utils import *
import pygame
import math


textModel = None
imageModel = None
dataset = None


def searchQuery(query, fontVectors):
    textData = Description.tokenizer([query], padding=False, return_tensors="pt")
    embeddedText = textModel(**textData).pooler_output

    fontIndices = []
    fontMatrix = None
    fontKeys = list(fontVectors).keys()
    for f, fontName in enumerate(fontKeys):
        fontIndices.extend([f] * len(fontVectors[fontName]))

        singleFontMatrix = torch.stack(fontVectors, dim=0)
        if fontMatrix is None:
            fontMatrix = singleFontMatrix
        else:
            fontMatrix = torch.cat([fontMatrix, singleFontMatrix], dim=0)

    fontIndices = torch.tensor(fontIndices, dtype=torch.long)
    scoreMatrix = fontMatrix @ embeddedText.t()
    fontScores = {}
    
    for f, fontName in enumerate(fontKeys):
        scores = scoreMatrix[fontIndices == f]
        fontScores[fontName] = {"median": torch.median(scores).item(),
                                "mean": torch.mean(scores).item(),
                                "std": torch.std(scores).item(),
                                "min": torch.min(scores).item(),
                                "max": torch.max(scores).item()}

    return fontScores


def generateFontVectors(batchSize=128):
    fontVectors = {}

    inputs = []
    names = []
    for i in range(len(dataset)):
        data = dataset[i]
        inputs.append(data["inputs"])
        names.append(data["name"])

    for i in range(0, len(inputs), batchSize):
        with torch.no_grad():
            j = min(len(inputs) - 1, i + batchSize)
            ips = torch.tensor(inputs[i:j], dtype=torch.float32).squeeze().unsqueeze(-1)
            outputs = imageModel(ips)

            batchNames = names[i:j]

            for n, batchName in enumerate(batchNames):
                if batchName not in fontVectors:
                    fontVectors[batchName] = []

                fontVectors[batchName].append(outputs[n])

    return fontVectors


def topKRankings(scores, rank, k=5):
    sortedScores = sorted(scores.items(), key=lambda item: item[1][rank])
    return sortedScores[:k]


queryText = ""
ranking = "mean"
rankingTypes = ["mean", "median", "min", "max", "std"]
currentRankings = {}
results = []
fontVectors = generateFontVectors()

pygame.init()
windowSize = [960, 640]
window = pygame.display.set_mode(windowSize)
clock = pygame.time.Clock()

uiFont = pygame.font.SysFont("Calibri", 12)
uiSurface = pygame.Surface(windowSize, pygame.SRCALPHA, 32)
uiSurface = uiSurface.convert_alpha()

displayCharacters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"
images = []
resetImages = False

searchBarWidth = int(960 * 3/4)
searchBarCorner = int(960 * 1/8), 96
resultsCorner = int(960 * 1/8), 160

while True:
    window.fill((55, 75, 200))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                ranking = rankingTypes[(rankingTypes.index(ranking) - 1) % len(rankingTypes)]
                results = topKRankings(currentRankings, ranking)
                resetImages = True
            elif event.key == pygame.K_RIGHT:
                ranking = rankingTypes[(rankingTypes.index(ranking) + 1) % len(rankingTypes)]
                results = topKRankings(currentRankings, ranking)
                resetImages = True
            elif event.key == pygame.K_BACKSPACE:
                queryText = queryText[:-1]
            elif event.key == pygame.K_RETURN:
                modifiedQuery = "a " + queryText + " font"
                currentRankings = searchQuery(modifiedQuery, fontVectors)
                results = topKRankings(currentRankings, ranking)
                resetImages = True
            else:
                queryText += event.unicode

    if resetImages:
        for result in results:
            name, score = result
            loadedFont = dataset.fonts[name]
            image = Image.Image()._new(loadedFont.getmask(displayCharacters))
            fontSurface = pygame.image.fromstring(image.tobytes(), image.size, image.mode).convert()
            images.append(fontSurface)

    pygame.draw.rect(uiSurface, [255, 255, 255], (*searchBarCorner, searchBarWidth, 32))
    queryRender = uiFont.render(queryText, False, [255, 255, 255])
    uiSurface.blit(queryRender, (searchBarCorner[0] + 8, searchBarCorner[1] + 4))

    height = 0
    for r, result in enumerate(results):
        fontImage = images[r]
        name, score = result

        nameRender = uiFont.render(f"{name} | {score}", False, [255, 255, 255])
        uiSurface.blit(nameRender, (resultsCorner[0], resultsCorner[1] + height))
        height += nameRender.get_height() + 8

        uiSurface.blit(fontImage, (resultsCorner[0], resultsCorner[1] + height))
        height += fontImage.get_height + 32

    window.blit(uiSurface, [0, 0])

    pygame.display.flip()
    clock.tick(60)
