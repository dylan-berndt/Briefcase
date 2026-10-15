import math
from PIL import Image
import numpy as np
import os
import cv2
from glob import glob
from .description import *
from .standard import scaleToFit
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed


def loadRochesterImage(args):
    imagePath, fontSize = args

    image = Image.open(imagePath).convert("RGB")
    array = np.array(image, dtype=np.float32)

    # Black glyph on white background → invert so glyph = 1.0
    gray = 1.0 - (array[:, :, 0] / 255.0)

    # tight crop + scale-to-fit, shared with the font-file render (standard.scaleToFit)
    canvas = scaleToFit(gray, fontSize)
    if canvas is None:
        return None, None

    name = os.path.basename(imagePath).removesuffix(".png")
    return name, canvas


def loadRochesterDescription(descriptionPath):
    name = os.path.basename(descriptionPath)
    text = open(descriptionPath, "r").read().split()

    description = Description(name, text)

    return name, description


def myFontsCacheMarker(directory, cacheSuffix):
    return os.path.join(directory, f"_cache_complete_smallimage{cacheSuffix}")


# cacheSuffix keeps several rendering resolutions side by side (see standard.collectFontSetPaths):
# "" is the original "smallimage" folder and its original "any .bmp present means done" check; a
# non-empty suffix uses "smallimage<suffix>" and only trusts an explicit completion marker, since
# an interrupted conversion would otherwise look finished forever.
def loadMyFontsImagePaths(directory, fontSize, cacheSuffix="", processes=30):
    print(f"\nLoading MyFonts images from {directory} {'=' * 20}")

    smallDir = "smallimage" + cacheSuffix
    if not os.path.exists(os.path.join(directory, smallDir)):
        os.mkdir(os.path.join(directory, smallDir))

    if cacheSuffix:
        needsConversion = not os.path.exists(myFontsCacheMarker(directory, cacheSuffix))
    else:
        needsConversion = len(glob(os.path.join(directory, smallDir, "*.bmp"))) == 0

    if needsConversion:
        imagePaths = glob(os.path.join(directory, "fontimage", "*.png"))
        tasks = [(path, fontSize) for path in imagePaths]
        with Pool(processes=processes) as pool:
            for i, (name, array) in enumerate(pool.imap(loadRochesterImage, tasks, chunksize=1000)):
                if name == None:
                    continue

                img = Image.fromarray((array * 255).astype(np.uint8)).convert('L')
                fontName = name.split("_")[0]
                letter = name[-1].lower()
                case = "u" if name[-1] == name[-2] else "l"
                imageName = f"{fontName} {letter}{case}.bmp"
                img.save(os.path.join(directory, smallDir, imageName))
                if i % 100 == 0:
                    print(f"\rImages converted: {i + 1}/{len(imagePaths)}", end="")

        if cacheSuffix:
            with open(myFontsCacheMarker(directory, cacheSuffix), "w") as marker:
                marker.write("complete\n")

    print()

    imagePaths = glob(os.path.join(directory, smallDir, "*.bmp"))
    names, letters, paths = [], [], []
    for p in imagePaths:
        name = os.path.basename(p).removesuffix(".bmp")
        # names end "<letter><l|u>": keep the case in the letter so al/au get different keys
        letter = name[-2].upper() if name[-1] == "u" else name[-2]
        names.append(name[:-3])
        letters.append(letter)
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