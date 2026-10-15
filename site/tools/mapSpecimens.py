"""The picture the maps show when the pointer is over a point: the point's own name, set in its own font file.

A map point is a font file ("Abel Bold"), so each gets its own tiny specimen. The label is "<family> <style>" as Pillow
reads it from the file, which is how the maps were labelled, so the label finds the file again. A few labels end in
" (google)" or " (dafont)" (the source of a name both had); that picks the source. Where several files share a label
(a .otf beside a .ttf, a variable font beside its statics) the family's preferred file is drawn.

Each image is one line tall, drawn at twice the size it is shown (the page shows it 22px tall) with the baseline in the
same place in every image, so fonts keep their relative sizes. A font that cannot draw its own name (a dingbat font,
say) shows its first few glyphs instead, followed by the name in a plain font. A font with nothing drawable, or one that
cannot be scaled, is left out and its point keeps the plain text label.

Output, per map, in site/backend/data/maps/:  <map>.pack (the images back to back) and <map>.index.json
({"count": points, "specimens": [[offset, length] or null per point, in the map's point order]}). The font files are only
read here, on the build machine; no font is ever sent to a browser.

    python site/tools/mapSpecimens.py
"""

import argparse
import io
import json
import math
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
REPO = os.path.dirname(SITE)
sys.path.append(HERE)
from fontList import preferredFile  # noqa: E402
from linkMaps import MAP_SOURCES, mapLabels  # noqa: E402
from specimens import supportedChars  # noqa: E402

HEIGHT = 44          # twice the 22px the page shows
SIZE = 28            # twice the page's 14px hover text
MIN_SIZE = 12
BASELINE = 32
PADDING = 8
MAX_WIDTH = 560      # twice 280px
GLYPHS = 12          # shown for a font that cannot draw its own name
INK = (232, 230, 224, 255)
MUTED = (122, 120, 112, 255)
ALPHA_QUALITY = 60
SUFFIX = re.compile(r" \((google|dafont)\)$")


def indexFiles(roots):
    """{label: [(source, path)]} for every font file under the {source: directory} roots."""
    byLabel = {}
    for source, root in roots.items():
        for directory, _, names in os.walk(root):
            for name in sorted(names):
                if not name.lower().endswith((".ttf", ".otf")):
                    continue
                path = os.path.join(directory, name)
                try:
                    family, style = ImageFont.truetype(path, 10).getname()
                except Exception:
                    continue
                byLabel.setdefault(f"{family} {style}", []).append((source, path))
    return byLabel


def resolve(label, sources, byLabel):
    """(file path, the name to draw) for a map label, or (None, name) when no file has it."""
    match = SUFFIX.search(label)
    name = label[:match.start()] if match else label
    allowed = {match.group(1)} & sources if match else sources
    paths = [path for source, path in byLabel.get(name, []) if source in allowed]
    return (preferredFile(paths) if paths else None), name


def drawable(chars):
    return [c for c in sorted(chars) if c.isprintable() and not c.isspace()]


def fitted(font, text, maxWidth):
    """The font at the largest size, down to MIN_SIZE, at which the text is no wider than maxWidth."""
    size = font.size
    while size > MIN_SIZE and font.getlength(text) > maxWidth:
        size -= 2
        font = font.font_variant(size=size)
    return font


def renderName(path, name):
    """WebP bytes of `name` in the font at `path`, or None when the font has nothing to show."""
    chars = supportedChars(path)
    letters = [c for c in name if not c.isspace()]
    try:
        font = ImageFont.truetype(path, SIZE)
    except OSError:
        return None        # e.g. a fixed-size bitmap font that cannot be scaled
    runs = []              # [(text, font, colour)]
    if letters and 2 * sum(c in chars for c in letters) >= len(letters):
        text = "".join(c for c in name if c in chars or c == " ").strip()
        runs.append((text, fitted(font, text, MAX_WIDTH - 2 * PADDING), INK))
    else:
        glyphs = "".join(drawable(chars)[:GLYPHS])
        if not glyphs:
            return None
        plain = ImageFont.load_default(size=SIZE)
        gap = font.getlength("  ")
        room = MAX_WIDTH - 2 * PADDING - gap
        glyphFont = fitted(font, glyphs, room * 0.6)
        nameFont = fitted(plain, name, room - glyphFont.getlength(glyphs))
        runs += [(glyphs, glyphFont, INK), (name, nameFont, MUTED)]
        gapRun = gap
    if not runs[0][0]:
        return None

    widths = [font.getlength(text) for text, font, _ in runs]
    spacing = gapRun if len(runs) > 1 else 0
    canvas = Image.new("RGBA", (math.ceil(sum(widths) + spacing) + 2 * PADDING, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    x = PADDING
    for (text, font, colour), width in zip(runs, widths):
        draw.text((x, BASELINE), text, font=font, fill=colour, anchor="ls")
        x += width + spacing
    out = io.BytesIO()
    canvas.save(out, format="WEBP", quality=80, alpha_quality=ALPHA_QUALITY, method=6)
    return out.getvalue()


def buildMap(labels, sources, byLabel, rendered=None):
    """(pack bytes, index, how many points have a specimen) for one map's labels. `rendered` caches the images by
    (file, name), so a file used by several maps is drawn once."""
    rendered = {} if rendered is None else rendered
    placed = {}
    pack = io.BytesIO()
    specimens = []
    for label in labels:
        path, name = resolve(label, sources, byLabel)
        if path is None:
            specimens.append(None)
            continue
        key = (path, name)
        if key not in rendered:
            rendered[key] = renderName(path, name)
        if rendered[key] is None:
            specimens.append(None)
            continue
        if key not in placed:
            placed[key] = [pack.tell(), len(rendered[key])]
            pack.write(rendered[key])
        specimens.append(placed[key])
    shown = sum(entry is not None for entry in specimens)
    return pack.getvalue(), {"count": len(labels), "specimens": specimens}, shown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=os.path.join(SITE, "frontend", "public", "maps"))
    parser.add_argument("--google", default=os.path.join(REPO, "google", "fonts"))
    parser.add_argument("--dafont", default=os.path.join(REPO, "dafont"))
    parser.add_argument("--out", default=os.path.join(SITE, "backend", "data", "maps"))
    args = parser.parse_args()

    roots = {"google": args.google, "dafont": args.dafont}
    byLabel = indexFiles({source: root for source, root in roots.items() if os.path.isdir(root)})
    print(f"{sum(len(v) for v in byLabel.values())} font files, {len(byLabel)} distinct names")
    os.makedirs(args.out, exist_ok=True)
    rendered = {}
    for name in sorted(os.listdir(args.maps)):
        if not name.endswith(".html"):
            continue
        stem = name[:-5]
        with open(os.path.join(args.maps, name), encoding="utf-8") as file:
            labels = mapLabels(file.read())
        pack, index, shown = buildMap(labels, MAP_SOURCES.get(stem, set(roots)), byLabel, rendered)
        with open(os.path.join(args.out, stem + ".pack"), "wb") as file:
            file.write(pack)
        with open(os.path.join(args.out, stem + ".index.json"), "w", encoding="utf-8") as file:
            json.dump(index, file, separators=(",", ":"))
        print(f"{stem}: {shown}/{len(labels)} points have a specimen, {len(pack) / 1e6:.1f} MB "
              f"({len(pack) / max(shown, 1) / 1e3:.1f} KB each)")


if __name__ == "__main__":
    main()
