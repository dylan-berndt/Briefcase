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


def imagesFromFont(fontData, fontSize, imageSize, save=None, chars=characters):
    font = ImageFont.truetype(fontData, fontSize)
    ascent, descent = font.getmetrics()
    # Gross way to do this, but I don't want another package
    standard = fontSize * (fontSize / ascent)
    font = ImageFont.truetype(fontData, standard)
    ttf = TTFont(fontData)
    badBox = font.getbbox('\uFFFF')

    fontName, fontStyle = font.getname()

    if save is not None:
        imagePath = os.path.join(save, "bitmaps", f"{fontName} {fontStyle} al.bmp")
        if os.path.exists(imagePath):
            return font, fontName, fontStyle, []

    canvases = []
    for char in chars:
        case = "l" if char == char.lower() else "u"
        name = f"{fontName} {fontStyle} {char.lower()}{case}"

        if save is not None:
            path = os.path.join(save, "bitmaps", name + ".bmp")
            if os.path.exists(path):
                continue

        hasGlyph = False
        for table in ttf['cmap'].tables:
            if ord(char) in table.cmap.keys():
                hasGlyph = True

        mask = font.getmask(char)
        box = font.getbbox(char)
        if mask.size == (0, 0) or not hasGlyph:
            continue
        im = Image.Image()._new(mask)

        canvas = Image.new("L", (imageSize, imageSize), 0)
        baseline = int(imageSize * 0.9)
        offset = baseline - ascent
        canvas.paste(im, ((imageSize - im.width) // 2, offset + box[1] - int(imageSize * 0.25)))
        if save is not None:
            canvas.save(path)
        else:
            canvases.append(canvas)

    return font, fontName, fontStyle, canvases


# TODO: Combine this and loadFontSet
def collectFontSetPaths(directory, fontSize, maps):
    print(f"\nCollecting font paths from {directory} {'=' * 20}")

    if not os.path.exists(os.path.join(directory, "bitmaps")):
        os.mkdir(os.path.join(directory, "bitmaps"))
    if not os.path.exists(os.path.join(directory, "sdf")):
        os.mkdir(os.path.join(directory, "sdf"))

    imageSize = int(fontSize * 1.5)
    ttfPaths = glob(os.path.join(directory, "fonts", "**", "*.ttf"), recursive=True)
    otfPaths = glob(os.path.join(directory, "fonts", "**", "*.otf"), recursive=True)

    for f, fontPath in enumerate(ttfPaths + otfPaths):
        if not os.path.isfile(fontPath):
            continue
        try:
            imagesFromFont(fontPath, fontSize, imageSize, directory)
        except Exception as e:
            print(fontPath, e)
        print(f"\rFonts serialized: {f + 1}/{len(ttfPaths + otfPaths)}", end="")

    print()

    if maps == "sdf":
        imagePaths = glob(os.path.join(directory, "bitmaps", "*"))
        for i, imagePath in enumerate(imagePaths):
            try:
                sdfPath = os.path.join(directory, "sdf",
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
    allPaths = glob(os.path.join(directory, maps, "*"))
    available = {os.path.basename(p): p for p in allPaths}

    names, letters, paths = [], [], []
    for basename, path in available.items():
        stem = basename.removesuffix(ext)
        char = stem[-2]

        if "ԵՒ" in stem:
            continue

        if char not in characters:
            continue

        names.append(stem[:-3])
        letters.append(char)
        paths.append(path)

    return {"names": np.array(names), "letters": np.array(letters), "paths": np.array(paths)}


def loadFontSet(directory, fontSize, maps):
    print(f"\nLoading font images from {directory} {'=' * 20}")

    if not os.path.exists(os.path.join(directory, "bitmaps")):
        os.mkdir(os.path.join(directory, "bitmaps"))

    if not os.path.exists(os.path.join(directory, "sdf")):
        os.mkdir(os.path.join(directory, "sdf"))

        # Just a bit of padding
    imageSize = int(fontSize * 1.5)
    ttfPaths = glob(os.path.join(directory, "fonts", "**", "*.ttf"), recursive=True)
    otfPaths = glob(os.path.join(directory, "fonts", "**", "*.otf"), recursive=True)
    fontPaths = ttfPaths + otfPaths

    fonts = {}
    fontMap = {}

    for f, fontPath in enumerate(fontPaths):
        if not os.path.isfile(fontPath):
            continue
        try:
            font, fontName, fontStyle, _ = imagesFromFont(fontPath, fontSize, imageSize, directory)
            fonts[f"{fontName} {fontStyle}"] = font
            fontMap[f"{fontName} {fontStyle}"] = fontName

        except Exception as e:
            print(fontPath, e)

        print(f"\rFonts serialized: {f + 1}/{len(fontPaths)}", end="")

    # Creating SDFs from all the bitmaps
    if maps == "sdf":
        imagePaths = glob(os.path.join(directory, "bitmaps", "*"))
        for i, imagePath in enumerate(imagePaths):
            try:
                path = os.path.join(directory, "sdf", os.path.basename(imagePath).removesuffix(".bmp") + ".npy")
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
    imagePaths = glob(os.path.join(directory, maps, "*"))
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