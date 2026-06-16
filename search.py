from utils import *
import pygame
import math


testDir = os.path.join("checkpoints", "finetune", "2026-06-07 17-04", "ViT openai-clip-vit-base-patch32")
_, conf = ViT.load(os.path.join("checkpoints", "pretrain", "latest"))
imageModel, conf = ViTEmbedder.load(testDir, model=ViT(conf.model), name="image")
textModel = CLIPTextEmbedder("openai/clip-vit-base-patch32", conf.model.embedDim)
textModel.load_state_dict(torch.load(os.path.join(testDir, "text.pt")))
Description.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

 
dataset = CombinedQueryData(conf.dataset, training=False)

textModel.eval()
imageModel.eval()

imageModel = imageModel.to(device)
textModel = textModel.to(device)


def searchQuery(query, fontEmbeddings):
    textData = Description.tokenizer([query], padding=False, return_tensors="pt")
    print(Description.tokenizer.convert_ids_to_tokens(textData["input_ids"][0]))
    textData = {k: v.to(device) for k, v in textData.items() if k != "token_type_ids"}
    embeddedText = textModel(textData).cpu()

    fontKeys = list(fontEmbeddings.keys())
    fontMatrix = torch.tensor(np.stack([fontEmbeddings[k] for k in fontKeys]), dtype=torch.float32)

    scores = fontMatrix @ nn.functional.normalize(embeddedText, dim=-1).t()
    return {fontKeys[i]: scores[i].item() for i in range(len(fontKeys))}


def topKRankings(scores, k=5):
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]


queryText = ""
currentRankings = {}
results = []
fontVectors = generateEmbeddings(
    {"names": dataset.names,
     "paths": dataset.paths,
     "letters": dataset.letters},
     model = imageModel,
     fileName = "allText"
)

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

fontPathMap = {}
for name, letter, path in zip(dataset.names, dataset.letters, dataset.paths):
    if name not in fontPathMap:
        fontPathMap[name] = {}
    fontPathMap[name][letter] = path

scrollOffset = 0


while True:
    window.fill((55, 75, 200))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                queryText = queryText[:-1]
            elif event.key == pygame.K_RETURN:
                currentRankings = searchQuery(queryText, fontVectors)
                results = topKRankings(currentRankings, k=100)
                print(len(results))
                resetImages = True
                scrollOffset = 0
            else:
                queryText += event.unicode

        if event.type == pygame.MOUSEWHEEL:
            scrollOffset = max(0, scrollOffset - event.y * 20)

    if resetImages:
        images = []
        for result in results:
            name, score = result
            letterMap = fontPathMap.get(name)
            if not letterMap:
                images.append(None)
                continue

            surfaces = []
            for char in displayCharacters:
                path = letterMap.get(char)
                if path is None:
                    continue
                _, arr = loadImage(path)
                if arr is None:
                    continue
                arr = np.squeeze(arr)
                gray = (arr * 255).astype(np.uint8)
                h, w = gray.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[..., :3] = 255
                rgba[..., 3] = gray
                surfaces.append(pygame.image.fromstring(rgba.tobytes(), (w, h), "RGBA"))

            if not surfaces:
                images.append(None)
                continue

            totalW = sum(s.get_width() for s in surfaces)
            maxH = max(s.get_height() for s in surfaces)
            combined = pygame.Surface((totalW, maxH), pygame.SRCALPHA)
            x = 0
            for surf in surfaces:
                combined.blit(surf, (x, 0))
                x += surf.get_width()
            images.append(combined)

        resetImages = False

    pygame.draw.rect(uiSurface, [255, 255, 255], (*searchBarCorner, searchBarWidth, 32))
    queryRender = uiFont.render(queryText, False, [0, 0, 0])
    uiSurface.blit(queryRender, (searchBarCorner[0] + 8, searchBarCorner[1] + 4))

    contentSurface = pygame.Surface(windowSize, pygame.SRCALPHA)
    height = -scrollOffset
    for r, result in enumerate(results):
        fontImage = images[r]
        name, score = result

        nameRender = uiFont.render(f"{name} | {score:.2f}", False, [255, 255, 255])
        contentSurface.blit(nameRender, (resultsCorner[0], height - 8))
        height += nameRender.get_height() - 16 + 8

        if fontImage is not None:
            contentSurface.blit(fontImage, (resultsCorner[0], height))
            height += fontImage.get_height() + 32

    scrollOffset = min(scrollOffset, max(0, height - (windowSize[1] - resultsCorner[1])))
    window.blit(contentSurface, (0, resultsCorner[1]))
    window.blit(uiSurface, [0, 0])

    pygame.display.flip()
    clock.tick(60)
