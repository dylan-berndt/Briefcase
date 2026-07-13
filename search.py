from utils import *
import pygame


# Swap to FontSearch() for the CLIP image/text embedding search.
search = TagSearch()


queryText = ""
results = []

pygame.init()
windowSize = [960, 640]
window = pygame.display.set_mode(windowSize)
clock = pygame.time.Clock()

uiFont = pygame.font.SysFont("Calibri", 24, bold=True)
uiSurface = pygame.Surface(windowSize, pygame.SRCALPHA, 32)
uiSurface = uiSurface.convert_alpha()

images = []
resetImages = False

searchBarWidth = int(960 * 3/4)
searchBarCorner = int(960 * 1/8), 96
resultsCorner = int(960 * 1/8), 160

scrollOffset = 0


def glyphSurface(name):
    strip = search.glyphStrip(name)
    if strip is None:
        return None
    h, w = strip.shape[:2]
    return pygame.image.fromstring(strip.tobytes(), (w, h), "RGBA")


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
                results = search.search(queryText, k=100)
                print(len(results))
                resetImages = True
                scrollOffset = 0
            else:
                queryText += event.unicode

        if event.type == pygame.MOUSEWHEEL:
            scrollOffset = max(0, scrollOffset - event.y * 20)

    if resetImages:
        images = [glyphSurface(name) for name, score in results]
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
