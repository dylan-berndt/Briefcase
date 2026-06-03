from utils import *
import pygame
import math


testDir = os.path.join("checkpoints", "finetune", "2026-05-29 15-32", "ViT openai-clip-vit-base-patch32")
model, conf = ViT.load(os.path.join("checkpoints", "pretrain", "latest"))
imageModel, conf = ViTEmbedder.load(testDir, model=model, name="image")
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
     fileName = "google"
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
                results = topKRankings(currentRankings)
                resetImages = True
            else:
                queryText += event.unicode

    if resetImages:
        images = []
        for result in results:
            # Wow I hate every image processing step here. This is stupid
            name, score = result
            if name not in dataset.fonts:
                images.append(None)
                continue
            loadedFont = dataset.fonts[name]
            if loadedFont is None:
                images.append(None)
                continue
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

        nameRender = uiFont.render(f"{name} | {score:.2f}", False, [255, 255, 255])
        window.blit(nameRender, (resultsCorner[0], resultsCorner[1] + height))
        height += nameRender.get_height() + 8

        if fontImage is not None:
            window.blit(fontImage, (resultsCorner[0], resultsCorner[1] + height))
            height += fontImage.get_height() + 32

    window.blit(uiSurface, [0, 0])

    pygame.display.flip()
    clock.tick(60)
