"""One specimen image per font: two lines of sample text set in the font itself, light ink on transparent so it sits
on the site's dark cards. Rendered from the font file on the build machine; the font is never sent to a browser."""

import io

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 640, 180
PADDING = 24
INK = (232, 230, 224, 255)
LINES = ["The quick brown fox jumps over the lazy dog", "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"]
MAX_SIZE, MIN_SIZE = 56, 10
# The alpha channel is most of the file; 60 is visibly identical on text and about 40% smaller than lossless
ALPHA_QUALITY = 60


def supportedChars(fontPath):
    try:
        cmap = TTFont(fontPath, lazy=True, fontNumber=0).getBestCmap() or {}
    except Exception:
        return set()
    return set(chr(code) for code in cmap)


def fitLine(fontPath, text, maxWidth, maxSize):
    """Largest font size (down to MIN_SIZE) at which the text fits maxWidth."""
    size = maxSize
    while size > MIN_SIZE:
        font = ImageFont.truetype(fontPath, size)
        if font.getlength(text) <= maxWidth:
            return font
        size -= 2
    return ImageFont.truetype(fontPath, MIN_SIZE)


def renderSpecimen(fontPath, quality=80):
    """WebP bytes, or None if the font has none of the sample characters."""
    chars = supportedChars(fontPath)
    lines = ["".join(c for c in line if c in chars or c == " ").strip() for line in LINES]
    lines = [line for line in lines if line]
    if not lines:
        return None

    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    # first line large, the rest at 60% of that size's ceiling
    fonts = [fitLine(fontPath, line, WIDTH - 2 * PADDING, MAX_SIZE if i == 0 else int(MAX_SIZE * 0.6))
             for i, line in enumerate(lines)]
    heights = [font.getmetrics()[0] + font.getmetrics()[1] for font in fonts]
    gap = 10
    y = (HEIGHT - sum(heights) - gap * (len(lines) - 1)) // 2
    for line, font, height in zip(lines, fonts, heights):
        draw.text((PADDING, max(y, 0)), line, font=font, fill=INK)
        y += height + gap

    out = io.BytesIO()
    canvas.save(out, format="WEBP", quality=quality, alpha_quality=ALPHA_QUALITY, method=6)
    return out.getvalue()
