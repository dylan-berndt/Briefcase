from utils import *
import pygame

searchHelper = MeanderSearch(backbone=os.path.join("checkpoints", "pretrain", "best"))

positive = None
negative = None
results = []

pygame.init()
windowSize = [960, 960]
window = pygame.display.set_mode(windowSize)
clock = pygame.time.Clock()

uiFont = pygame.font.SysFont("Calibri", 24, bold=True)
uiSurface = pygame.Surface(windowSize, pygame.SRCALPHA, 32)
uiSurface = uiSurface.convert_alpha()

images = []
resetImages = False

searchBarCorner = int(960 * 1/8), 96
resultsCorner = int(960 * 1/8), 480

scrollOffset = 0


def glyphSurface(name):
    strip = searchHelper.glyphStrip(name)
    if strip is None:
        return None
    h, w = strip.shape[:2]
    return pygame.image.fromstring(strip.tobytes(), (w, h), "RGBA")


optionImages = [glyphSurface(name) for name, embedding in searchHelper.options]


while True:
    window.fill((55, 75, 200))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                height = 0
                selected = None
                for o, option in enumerate(searchHelper.options):
                    fontImage = optionImages[o]
                    if fontImage is not None:
                        imageHeight = fontImage.get_height()
                        if height < pygame.mouse.get_pos()[1] - searchBarCorner[1] < height + imageHeight:
                            selected = option
                            break
                        height += imageHeight + 32

                if selected is not None:
                    if positive is None:
                        positive = selected
                    else:
                        negative = selected

            if positive is not None and negative is not None:
                searchHelper.updateLocation(positive, negative)
                results = searchHelper.search(None, k=100)
                print(len(results))
                resetImages = True
                scrollOffset = 0
                positive = None
                negative = None

        if event.type == pygame.MOUSEWHEEL:
            scrollOffset = max(0, scrollOffset - event.y * 20)

    if resetImages:
        images = [glyphSurface(name) for name, score in results]
        optionImages = [glyphSurface(name) for name, embedding in searchHelper.options]
        resetImages = False


    uiSurface.fill((0, 0, 0, 0))

    height = 0
    for o, option in enumerate(searchHelper.options):
        fontImage = optionImages[o]
        if fontImage is not None:
            uiSurface.blit(fontImage, (searchBarCorner[0], height))
            height += fontImage.get_height() + 32


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
