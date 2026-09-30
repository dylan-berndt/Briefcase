"""Tiny TrueType fonts built with fontTools, so the tools can be tested on files whose look is known."""

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = LOWER.upper()
DIGITS = "0123456789"


def rectangle(width, height):
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0))
    pen.lineTo((100, height))
    pen.lineTo((100 + width, height))
    pen.lineTo((100 + width, 0))
    pen.closePath()
    return pen.glyph()


def makeFont(path, chars, width=600, height=700, family="Synth"):
    """Every character is a width x height rectangle; chars without an entry are missing from the cmap."""
    order = [".notdef"] + [f"g{ord(c)}" for c in chars]
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({ord(c): f"g{ord(c)}" for c in chars})
    glyphs = {name: rectangle(width, height) for name in order}
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (800, 100) for name in order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": family, "styleName": "Regular"})
    builder.setupOS2(sTypoAscender=800, usWinAscent=800, usWinDescent=200)
    builder.setupPost()
    builder.save(str(path))
    return str(path)
