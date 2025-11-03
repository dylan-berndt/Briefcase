from utils import *
import pygame
import math


testDir = os.path.join("checkpoints", "finetune", "latest", "upper bert-base-uncased")
textModel = BertModel.from_pretrained(os.path.join(testDir, "text"))
Description.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

imageModel, conf = UNet.load(testDir, name="image")
conf.dataset.directory = "google"
dataset = FontData(conf.dataset, training=False)

textModel.eval()
imageModel.eval()


def searchQuery(query, fontVectors):
    textData = Description.tokenizer([query], padding=False, return_tensors="pt")
    embeddedText = textModel(**textData).pooler_output

    fontIndices = []
    fontMatrix = None
    fontKeys = list(fontVectors.keys())
    for f, fontName in enumerate(fontKeys):
        fontIndices.extend([f] * len(fontVectors[fontName]))

        singleFontMatrix = torch.stack(fontVectors[fontName], dim=0)
        if fontMatrix is None:
            fontMatrix = singleFontMatrix
        else:
            fontMatrix = torch.cat([fontMatrix, singleFontMatrix], dim=0)

    fontIndices = torch.tensor(fontIndices, dtype=torch.long)
    f, e = nn.functional.normalize(fontMatrix, dim=-1), nn.functional.normalize(embeddedText, dim=-1)
    scoreMatrix = f @ e.t()
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
    testCharacters = [chr(c) for c in latin]

    fontVectors = {}

    inputs = []
    names = []
    letters = []
    for i in range(len(dataset)):
        data = dataset[i]
        inputs.append(data["inputs"].cpu())
        names.append(data["name"])
        letters.append(data["letter"])

    inputs = np.array(inputs)
    names = np.array(names)
    letters = np.array(letters)

    mask = np.isin(letters, np.array(testCharacters))
    inputs = inputs[mask]
    names = names[mask]

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

        print(f"\r{i}/{len(inputs)} images processed", end="")

    return fontVectors


def topKRankings(scores, rank, k=5):
    sortedScores = sorted(scores.items(), key=lambda item: item[1][rank], reverse=True)
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

uiFont = pygame.font.SysFont("Calibri", 24, bold=True)
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
        images = []
        for result in results:
            # Wow I hate every image processing step here. This is stupid
            name, score = result
            loadedFont = dataset.fonts[name]
            size = loadedFont.getmask(displayCharacters).size
            image = Image.new("RGBA", size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), displayCharacters, font=loadedFont, fill=(255, 255, 255, 255))

            fontSurface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
            images.append(fontSurface)

        resetImages = False

    pygame.draw.rect(uiSurface, [255, 255, 255], (*searchBarCorner, searchBarWidth, 32))
    queryRender = uiFont.render(queryText, False, [0, 0, 0])
    uiSurface.blit(queryRender, (searchBarCorner[0] + 8, searchBarCorner[1] + 4))

    height = 0
    for r, result in enumerate(results):
        fontImage = images[r]
        name, score = result

        nameRender = uiFont.render(f"{name} | {ranking}: {score[ranking]:.2f}", False, [255, 255, 255])
        window.blit(nameRender, (resultsCorner[0], resultsCorner[1] + height))
        height += nameRender.get_height() + 8

        window.blit(fontImage, (resultsCorner[0], resultsCorner[1] + height))
        height += fontImage.get_height() + 32

    window.blit(uiSurface, [0, 0])

    pygame.display.flip()
    clock.tick(60)
