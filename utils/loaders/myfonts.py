import math
from PIL import Image
import numpy as np
import os
import cv2
from glob import glob
from .description import *
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed


def rochesterImageLoaderFactory(fontSize):
    def loadRochesterImage(self, imagePath):
        image = Image.open(imagePath).convert("RGBA")
        array = np.array(image)

        array = 1 - (array[:, :, 0] / 255)
        charWidth = np.argmax(np.arange(array.shape[1]) * np.max(array, axis=0))
        alpha = array[:, :charWidth]

        if alpha.shape[0] == 0 or alpha.shape[1] == 0:
            return None, None

        # Height / Width
        ratio = alpha.shape[0] / alpha.shape[1]
        width, height = int(self.fontSize / ratio), self.fontSize

        if height <= 0 or width <= 0:
            return None, None

        fixed = cv2.resize(alpha, [height, width])

        imageSize = int(self.fontSize * 1.5)
        overflow = math.ceil((fixed.shape[1] - imageSize) / 2)
        if overflow > 0:
            fixed = fixed[:, overflow: overflow + imageSize]
        overflow = math.ceil((fixed.shape[0] - imageSize) / 2)
        if overflow > 0:
            fixed = fixed[overflow: overflow + imageSize]

        full = np.zeros([imageSize, imageSize], dtype=np.float32)

        hPad = (imageSize - fixed.shape[0]) // 2
        wPad = (imageSize - fixed.shape[1]) // 2
        full[hPad:hPad + fixed.shape[0], wPad:wPad + fixed.shape[1]] = fixed

        name = os.path.basename(imagePath).removesuffix(".png")

        return name, full
    
    return loadRochesterImage


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
        imageFunc = rochesterImageLoaderFactory(fontSize)
        imagePaths = glob(os.path.join(directory, "fontimage", "*.png"))
        with Pool(processes=2) as pool:
            for i, (name, array) in enumerate(pool.imap(imageFunc, imagePaths, chunksize=1000)):
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