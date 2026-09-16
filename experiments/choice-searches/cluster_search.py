from utils import *
from utils.search import DISPLAY_CHARACTERS
import pygame


# ClusteringSearch narrows the corpus by assigning each font a k-ary "code":
# each digit position splits the remaining fonts into k clusters. Pick the
# cluster containing your target font, one digit at a time. Press Backspace
# to undo the last digit (reselect it), or R to reset back to digit 1.
search = ClusteringSearch(k=8, digits=8, visible=40)

resetImages = False
images = {}
scaledImages = {}

pygame.init()
windowSize = [1400, 800]
window = pygame.display.set_mode(windowSize)
clock = pygame.time.Clock()

uiFont = pygame.font.SysFont("Calibri", 24, bold=True)
smallFont = pygame.font.SysFont("Calibri", 18)

headerCorner = 40, 24
optionsCorner = 40, 88

scrollOffset = 0

COLUMNS = 3
COLUMN_WIDTH = (windowSize[0] - optionsCorner[0]) // COLUMNS

# Full A-Z/a-z strip is way too wide to shrink into a grid cell without
# crushing every glyph down to a few pixels tall, so thumbnails only render
# a short recognizable sample; the final matched-fonts view still uses the
# full strip since it isn't squeezed into a column.
PREVIEW_CHARACTERS = "AaGgQq"
PREVIEW_HEIGHT = 48
RESULT_HEIGHT = 64


def glyphSurface(name, characters=DISPLAY_CHARACTERS):
    cacheKey = (name, characters)
    if cacheKey in images:
        return images[cacheKey]
    strip = search.glyphStrip(name, characters=characters)
    surface = None
    if strip is not None:
        h, w = strip.shape[:2]
        surface = pygame.image.fromstring(strip.tobytes(), (w, h), "RGBA")
    images[cacheKey] = surface
    return surface


def scaledGlyphSurface(name, characters, targetHeight, maxWidth=None):
    cacheKey = (name, characters, targetHeight, maxWidth)
    if cacheKey in scaledImages:
        return scaledImages[cacheKey]

    surface = glyphSurface(name, characters=characters)
    if surface is None:
        scaledImages[cacheKey] = None
        return None

    scale = targetHeight / surface.get_height()
    if maxWidth is not None and surface.get_width() * scale > maxWidth:
        scale = maxWidth / surface.get_width()

    size = (max(1, int(surface.get_width() * scale)), max(1, int(surface.get_height() * scale)))
    scaled = pygame.transform.smoothscale(surface, size)
    scaledImages[cacheKey] = scaled
    return scaled


def chooseDigit(value):
    search.updateLocation(value)


def undo():
    search.undo()


def reset():
    search.reset()


while True:
    window.fill((30, 30, 45))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                undo()
                scrollOffset = 0
            elif event.key == pygame.K_r:
                reset()
                scrollOffset = 0
            elif not search.finished and pygame.K_1 <= event.key <= pygame.K_9:
                digitChoice = event.key - pygame.K_1
                if digitChoice < search.k:
                    chooseDigit(digitChoice)
                    scrollOffset = 0

        if event.type == pygame.MOUSEWHEEL:
            scrollOffset = max(0, scrollOffset - event.y * 20)

    contentSurface = pygame.Surface(windowSize, pygame.SRCALPHA)
    height = -scrollOffset

    if search.finished:
        matched = search.matchedNames()
        header = f"Code fully assigned ({search.digits} digits) -- {len(matched)} font(s) match. " \
                 f"[Backspace] reselect last digit  [R] reset"
        headerRender = uiFont.render(header, False, [255, 255, 255])
        window.blit(headerRender, headerCorner)

        maxStripWidth = windowSize[0] - optionsCorner[0] - 16
        for name in matched:
            nameRender = uiFont.render(name, False, [255, 255, 255])
            contentSurface.blit(nameRender, (optionsCorner[0], height))
            height += nameRender.get_height() + 4

            fontImage = scaledGlyphSurface(name, DISPLAY_CHARACTERS, RESULT_HEIGHT, maxWidth=maxStripWidth)
            if fontImage is not None:
                contentSurface.blit(fontImage, (optionsCorner[0], height))
                height += fontImage.get_height() + 24
    else:
        header = f"Digit {search.digit + 1} of {search.digits} -- press 1-{search.k} to pick a cluster. " \
                 f"[Backspace] reselect last digit  [R] reset"
        headerRender = uiFont.render(header, False, [255, 255, 255])
        window.blit(headerRender, headerCorner)

        # Masonry-style stacking: each option's height depends on how many
        # fonts are actually in it, so track a running height per column
        # instead of assuming a fixed row height.
        columnHeights = [-scrollOffset] * COLUMNS
        maxPreviewWidth = COLUMN_WIDTH - 24
        for i, cluster in enumerate(search.options):
            col = i % COLUMNS
            x = optionsCorner[0] + col * COLUMN_WIDTH
            y = columnHeights[col]

            labelRender = uiFont.render(f"[{i + 1}] {len(cluster)} shown", False, [255, 220, 130])
            contentSurface.blit(labelRender, (x, y))
            itemHeight = labelRender.get_height() + 8

            for name, _ in cluster:
                nameRender = smallFont.render(name, False, [255, 255, 255])
                contentSurface.blit(nameRender, (x, y + itemHeight))
                itemHeight += nameRender.get_height() + 2

                fontImage = scaledGlyphSurface(name, PREVIEW_CHARACTERS, PREVIEW_HEIGHT, maxWidth=maxPreviewWidth)
                if fontImage is not None:
                    contentSurface.blit(fontImage, (x, y + itemHeight))
                    itemHeight += fontImage.get_height() + 10

            columnHeights[col] = y + itemHeight + 24

        height = max(h + scrollOffset for h in columnHeights) if columnHeights else 0

    scrollOffset = min(scrollOffset, max(0, height - (windowSize[1] - optionsCorner[1])))
    window.blit(contentSurface, (0, optionsCorner[1]))

    pygame.display.flip()
    clock.tick(60)
