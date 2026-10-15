import hashlib
import os
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from glob import glob
from scipy.ndimage import distance_transform_edt as dist
from concurrent.futures import ThreadPoolExecutor, as_completed

latin = list(range(ord('a'), ord('z') + 1))
greek = list(range(0x03B1, 0x03C9 + 1))
armenian = list(range(0x0561, 0x0587 + 1))
cyrillic = list(range(0x0430, 0x044F + 1))
georgian = list(range(0x10D0, 0x10FF + 1))
characters = latin + greek + armenian + cyrillic
# characters = latin
characters = [chr(c) for c in characters]
characters = characters + [c.upper() for c in characters]

# 2 characters technically, and messes with checks for empty glyphs
characters.remove("ԵՒ")

FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")
RENDER_PX = 200        # glyphs are rendered large, then tight-cropped and scaled to fit (MyFonts pipeline)

# Cache folders/markers for the standardized MyFonts-style render with hashed font keys. The old metric-render
# caches ("bitmaps", "sdf", "_cache_complete") are left alone and no longer read: their file names lack the hash
# and their glyphs come from a different render, so mixing them in would silently double every font.
CACHE_VERSION = "_mf"


def bitmapDirName(cacheSuffix=""):
    return "bitmaps" + CACHE_VERSION + cacheSuffix


def sdfDirName(cacheSuffix=""):
    return "sdf" + CACHE_VERSION + cacheSuffix


def scaleToFit(gray, fontSize):
    """The MyFonts dataset's glyph normalization (gray: glyph = 1 on 0): trim the empty columns on the right and the
    empty rows, scale to fit a (imageSize - 16)px box keeping the aspect ratio, centre on an imageSize canvas.
    loadRochesterImage (myfonts.py) uses this on the MyFonts PNGs and imagesFromFont on font-file renders, so every
    source goes through the same normalization."""
    imageSize = int(fontSize * 1.5)
    padding = 8
    targetGlyphSize = imageSize - 2 * padding

    if gray.size == 0:
        return None

    # Trim the large right-side whitespace: last col where any pixel > threshold
    col_max = np.max(gray, axis=0)
    nonzero_cols = np.where(col_max > 0.05)[0]
    if len(nonzero_cols) == 0:
        return None
    gray = gray[:, : nonzero_cols[-1] + 1]

    # Tight crop rows
    row_max = np.max(gray, axis=1)
    nonzero_rows = np.where(row_max > 0.05)[0]
    if len(nonzero_rows) == 0:
        return None
    gray = gray[nonzero_rows[0]: nonzero_rows[-1] + 1, :]

    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return None

    # Scale to fit within targetGlyphSize preserving aspect ratio
    h, w = gray.shape
    scale = min(targetGlyphSize / h, targetGlyphSize / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interp)

    # Center in canvas (glyph=1.0, background=0.0)
    canvas = np.zeros((imageSize, imageSize), dtype=np.float32)
    y0 = (imageSize - new_h) // 2
    x0 = (imageSize - new_w) // 2
    canvas[y0: y0 + new_h, x0: x0 + new_w] = resized
    return canvas


def fileHash(fontPath, length=12):
    with open(fontPath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:length]


def fontKey(font, fontPath):
    """"<family> <style> <sha256[:12]>": the name keeps prefix matching against descriptions working, the hash keeps
    two different files with the same internal name apart and gives byte-identical copies the same key."""
    fontName, fontStyle = font.getname()
    return f"{fontName} {fontStyle} {fileHash(fontPath)}"


def openFont(fontPath):
    """(PIL font at RENDER_PX, the cmaps of the first face). .ttc collections use their first face."""
    font = ImageFont.truetype(fontPath, RENDER_PX)
    ttf = TTFont(fontPath, lazy=True, fontNumber=0)
    return font, [table.cmap for table in ttf["cmap"].tables]


def renderGlyph(font, cmaps, char, fontSize):
    """One glyph on the standardized render (float32, glyph = 1), or None when the font doesn't map the character
    (PIL would otherwise draw the .notdef box) or draws nothing."""
    if not any(ord(char) in cmap for cmap in cmaps):
        return None
    mask = font.getmask(char)
    if mask.size == (0, 0):
        return None
    return scaleToFit(np.asarray(Image.Image()._new(mask), dtype=np.float32) / 255.0, fontSize)


def toUint8(canvas):
    """The truncating conversion the cached .bmp files have always used."""
    return (canvas * 255).astype(np.uint8)


# Lets us choose what kind of images to train on
def loadImage(imagePath):
    suffix = imagePath.split(".")[-1]
    try:
        if suffix == "npy":
            name, image = os.path.basename(imagePath).removesuffix(".npy"), np.load(imagePath)
        else:
            img = np.fromfile(imagePath, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            name, image = os.path.basename(imagePath).removesuffix(".bmp"), img
    except Exception as e:
        return None, None

    return name, image


# Not rewriting it correctly who give a damn
def imageWrapper(args):
    try:
        return imagesFromFont(*args)
    except Exception as e:
        print(args[0], e)


def imagesFromFont(fontData, fontSize, imageSize, save=None, chars=characters, bitmapDir=None):
    """Renders a font file's glyphs with the standardized MyFonts-style render (renderGlyph). With `save`, writes
    "<fontKey> <char><l|u>.bmp" into save/bitmapDir and skips glyphs already there; otherwise returns the canvases.
    imageSize is int(fontSize * 1.5) and kept only for the existing call signature."""
    if not os.path.isfile(fontData):
        return
    bitmapDir = bitmapDir or bitmapDirName()

    font, cmaps = openFont(fontData)
    key = fontKey(font, fontData)
    fontName, fontStyle = font.getname()

    if save is not None:
        imagePath = os.path.join(save, bitmapDir, f"{key} al.bmp")
        if os.path.exists(imagePath):
            return font, fontName, fontStyle, []

    canvases = []
    for char in chars:
        case = "l" if char == char.lower() else "u"
        name = f"{key} {char.lower()}{case}"

        if save is not None:
            path = os.path.join(save, bitmapDir, name + ".bmp")
            if os.path.exists(path):
                continue

        canvas = renderGlyph(font, cmaps, char, fontSize)
        if canvas is None:
            continue
        image = Image.fromarray(toUint8(canvas))
        if save is not None:
            image.save(path)
        else:
            canvases.append(image)

    return font, fontName, fontStyle, canvases


def fontFilePaths(directory):
    """Every font file under <directory>/fonts, in sorted order."""
    return sorted(os.path.join(root, name) for root, _, names in os.walk(os.path.join(directory, "fonts"))
                  for name in names if name.lower().endswith(FONT_EXTENSIONS))


def cacheCompleteMarker(directory, cacheSuffix):
    """Marker written by experiments/embedding-geometry/build_bitmap_cache.py once a suffixed
    cache is fully built. Lives in `directory`, NOT inside the bitmap folder (that folder is
    globbed and every file in it is parsed as a glyph)."""
    return os.path.join(directory, f"_cache_complete{CACHE_VERSION}{cacheSuffix}")


# TODO: Combine this and loadFontSet
# cacheSuffix lets several rendering resolutions coexist. Cached bitmap filenames encode font,
# style, glyph and case but NOT canvas size, and the "already rendered" checks key on filename
# alone, so a different fontSize into the same folder would silently reuse stale images.
# Folders are bitmapDirName(cacheSuffix) / sdfDirName(cacheSuffix).
def collectFontSetPaths(directory, fontSize, maps, cacheSuffix=""):
    print(f"\nCollecting font paths from {directory} {'=' * 20}")

    bitmapDir = bitmapDirName(cacheSuffix)
    sdfDir = sdfDirName(cacheSuffix)
    mapsDir = sdfDir if maps == "sdf" else bitmapDir

    if not os.path.exists(os.path.join(directory, bitmapDir)):
        os.mkdir(os.path.join(directory, bitmapDir))
    if not os.path.exists(os.path.join(directory, sdfDir)):
        os.mkdir(os.path.join(directory, sdfDir))

    imageSize = int(fontSize * 1.5)
    fontPaths = fontFilePaths(directory)
    if os.path.exists(cacheCompleteMarker(directory, cacheSuffix)):
        # fully built by build_bitmap_cache.py: skip the per-font-file open/parse pass, which
        # costs ~20 minutes per launch even when every bitmap already exists
        fontPaths = []

    # tasks = [(path, fontSize, imageSize, directory) for path in (ttfPaths + otfPaths)]

    # with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
    #     for i, result in enumerate(executor.map(imageWrapper, tasks)):
    #         if i % 100 == 0:
    #             print(f"\rFonts serialized: {i + 1}/{len(ttfPaths + otfPaths)}", end="")

    for f, fontPath in enumerate(fontPaths):
        if not os.path.isfile(fontPath):
            continue
        try:
            imagesFromFont(fontPath, fontSize, imageSize, directory, bitmapDir=bitmapDir)
        except Exception as e:
            message = f"{fontPath!r} {e!r}"
            print(message.encode("ascii", "backslashreplace").decode("ascii"))
        print(f"\rFonts serialized: {f + 1}/{len(fontPaths)}", end="")

    print()

    if maps == "sdf":
        imagePaths = glob(os.path.join(directory, bitmapDir, "*"))
        for i, imagePath in enumerate(imagePaths):
            try:
                sdfPath = os.path.join(directory, sdfDir,
                              os.path.basename(imagePath).removesuffix(".bmp") + ".npy")
                if os.path.exists(sdfPath):
                    continue
                img = np.fromfile(imagePath, dtype=np.float32)
                bits = img > (np.max(img) - np.min(img)) / 2
                sdf = dist(bits) - dist(~bits)
                np.save(sdfPath, sdf / imageSize)
            except Image.UnidentifiedImageError:
                print(imagePath, "Unidentified")
            print(f"\rSDFs generated: {i + 1}/{len(imagePaths)}", end="")

    print()

    ext = ".npy" if maps == "sdf" else ".bmp"

    # Build a set of available filenames for quick sibling lookup
    allPaths = glob(os.path.join(directory, mapsDir, "*"))
    available = {os.path.basename(p): p for p in allPaths}

    names, letters, paths = [], [], []
    for basename, path in available.items():
        stem = basename.removesuffix(ext)
        # filenames end "<char.lower()><l|u>" (see imagesFromFont): the last character is the case, the
        # one before it the glyph. The letter keeps its case so the lowercase and uppercase glyph of a
        # font get different (name, letter) keys instead of overwriting each other in dict lookups.
        case, char = stem[-1], stem[-2]

        if "ԵՒ" in stem:
            continue

        if case not in ("l", "u"):
            continue

        if case == "u":
            char = char.upper()

        if char not in characters:
            continue

        names.append(stem[:-3])
        letters.append(char)
        paths.append(path)

    return {"names": np.array(names), "letters": np.array(letters), "paths": np.array(paths)}


def loadFontSet(directory, fontSize, maps):
    print(f"\nLoading font images from {directory} {'=' * 20}")

    bitmapDir, sdfDir = bitmapDirName(), sdfDirName()
    os.makedirs(os.path.join(directory, bitmapDir), exist_ok=True)
    os.makedirs(os.path.join(directory, sdfDir), exist_ok=True)

        # Just a bit of padding
    imageSize = int(fontSize * 1.5)
    fontPaths = fontFilePaths(directory)

    fonts = {}
    fontMap = {}

    for f, fontPath in enumerate(fontPaths):
        if not os.path.isfile(fontPath):
            continue
        try:
            font, fontName, fontStyle, _ = imagesFromFont(fontPath, fontSize, imageSize, directory)
            key = fontKey(font, fontPath)
            fonts[key] = font
            fontMap[key] = fontName

        except Exception as e:
            print(fontPath, e)

        print(f"\rFonts serialized: {f + 1}/{len(fontPaths)}", end="")

    # Creating SDFs from all the bitmaps
    if maps == "sdf":
        imagePaths = glob(os.path.join(directory, bitmapDir, "*"))
        for i, imagePath in enumerate(imagePaths):
            try:
                path = os.path.join(directory, sdfDir, os.path.basename(imagePath).removesuffix(".bmp") + ".npy")
                if os.path.exists(path):
                    continue

                img = np.fromfile(imagePath, dtype=np.float32)

                # Magic for sdf generation
                bits = img > (np.max(img) - np.min(img)) / 2
                sdf = dist(bits) - dist(~bits)

                np.save(path, sdf / imageSize)
            except Image.UnidentifiedImageError:
                print(imagePath, "Unidentified")

            print(f"\rSDFs generated: {i + 1}/{len(imagePaths)}", end="")

    images = {}
    imagePaths = glob(os.path.join(directory, sdfDir if maps == "sdf" else bitmapDir, "*"))
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        for i, result in enumerate(executor.map(loadImage, imagePaths)):
            name, image = result
            images[name] = image

            if i % 100 == 0:
                print(f"\rImages loaded: {i + 1}/{len(imagePaths)}", end="")

    pairs = []
    mse = []
    letters = []
    names = []
    for key in images:
        case = key[-1]
        if case == "l":
            continue
        other = key[:-1] + "l"

        if other not in images:
            continue

        if key[-2] not in characters:
            continue

        mse.append(np.mean(np.power(images[other] - images[key], 2)))

        pairs.append((images[other], images[key]))
        letters.append(key[-2])

        names.append(key[:-3])

    names = np.array(names)
    pairs = np.array(pairs)
    mse = np.array(mse)
    letters = np.array(letters)

    return {"names": names, "letters": letters, "pairs": pairs, "mse": mse}


def loadImageDataset(config):
    pass