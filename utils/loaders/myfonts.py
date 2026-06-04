import math
from PIL import Image
import numpy as np
import os
import cv2
from glob import glob
from .description import *
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed


def loadRochesterImage(args):
    imagePath, fontSize = args
    imageSize = int(fontSize * 1.5)
    padding = 8
    targetGlyphSize = imageSize - 2 * padding

    image = Image.open(imagePath).convert("RGB")
    array = np.array(image, dtype=np.float32)

    # Black glyph on white background → invert so glyph = 1.0
    gray = 1.0 - (array[:, :, 0] / 255.0)

    # Trim the large right-side whitespace: last col where any pixel > threshold
    col_max = np.max(gray, axis=0)
    nonzero_cols = np.where(col_max > 0.05)[0]
    if len(nonzero_cols) == 0:
        return None, None
    gray = gray[:, : nonzero_cols[-1] + 1]

    # Tight crop rows
    row_max = np.max(gray, axis=1)
    nonzero_rows = np.where(row_max > 0.05)[0]
    if len(nonzero_rows) == 0:
        return None, None
    gray = gray[nonzero_rows[0]: nonzero_rows[-1] + 1, :]

    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return None, None

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

    name = os.path.basename(imagePath).removesuffix(".png")
    return name, canvas


def loadRochesterDescription(descriptionPath):
    name = os.path.basename(descriptionPath)
    text = open(descriptionPath, "r").read().split()

    description = Description(name, text)

    return name, description


def loadMyFontsImagePaths(directory, fontSize):
    print(f"\nLoading MyFonts images from {directory} {'=' * 20}")

    if not os.path.exists(os.path.join(directory, "smallimage")):
        os.mkdir(os.path.join(directory, "smallimage"))

    if len(glob(os.path.join(directory, "smallimage", "*.bmp"))) == 0:
        imagePaths = glob(os.path.join(directory, "fontimage", "*.png"))
        tasks = [(path, fontSize) for path in imagePaths]
        with Pool(processes=30) as pool:
            for i, (name, array) in enumerate(pool.imap(loadRochesterImage, tasks, chunksize=1000)):
                if name == None:
                    continue

                img = Image.fromarray((array * 255).astype(np.uint8)).convert('L')
                fontName = name.split("_")[0]
                letter = name[-1].lower()
                case = "u" if name[-1] == name[-2] else "l"
                imageName = f"{fontName} {letter}{case}.bmp"
                img.save(os.path.join(directory, "smallimage", imageName))
                if i % 100 == 0:
                    print(f"\rImages converted: {i + 1}/{len(imagePaths)}", end="")

    print()

    imagePaths = glob(os.path.join(directory, "smallimage", "*.bmp"))
    names, letters, paths = [], [], []
    for p in imagePaths:
        name = os.path.basename(p).removesuffix(".bmp")
        names.append(name[:-3])
        letters.append(name[-2])
        paths.append(p)

    return {"names": np.array(names), "letters": np.array(letters), "paths": np.array(paths)}


def loadRochesterDescriptions(directory):
    descriptions = {}
    descriptionPaths = glob(os.path.join(directory, "taglabel", "*"))
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        futures = {executor.submit(loadRochesterDescription, p): p for p in descriptionPaths}
        for i, future in enumerate(as_completed(futures)):
            results = future.result()
            name, description = results
            descriptions[name] = description

            if i % 100 == 0:
                print(f"\rDescriptions loaded: {i + 1}/{len(descriptionPaths)}", end="")

    print()

    return descriptions