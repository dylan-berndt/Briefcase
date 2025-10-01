import torch
from torch.utils.data import Dataset, DataLoader, Subset
from .config import *
import numpy as np

import os
from glob import glob
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt as dist
from PIL import Image, ImageFont

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

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


class FontData(Dataset):
    def __init__(self, config: Config):
        if not os.path.exists(os.path.join(config.directory, "bitmaps")):
            os.mkdir(os.path.join(config.directory, "bitmaps"))

        if not os.path.exists(os.path.join(config.directory, "sdf")):
            os.mkdir(os.path.join(config.directory, "sdf"))

         # Just a bit of padding
        imageSize = int(config.fontSize * 1.5)
        ttfPaths = glob(os.path.join(config.directory, "fonts", "**", "*.ttf"), recursive=True)
        otfPaths = glob(os.path.join(config.directory, "fonts", "**", "*.otf"), recursive=True)
        fontPaths = ttfPaths + otfPaths
        for f, fontPath in enumerate(fontPaths):
            if not os.path.isfile(fontPath):
                continue
            try:
                font = ImageFont.truetype(fontPath, config.fontSize)
                ascent, descent = font.getmetrics()
                # Gross way to do this, but I don't want another package
                standard = config.fontSize * (config.fontSize / ascent)
                font = ImageFont.truetype(fontPath, standard)
                badBox = font.getbbox('\uFFFF')
                for char in characters:
                    case = "l" if char == char.lower() else "u"
                    fontName, fontStyle = font.getname()
                    name = f"{fontName} {fontStyle} {char.lower()}{case}"
                    path = os.path.join(config.directory, "bitmaps", name + ".bmp")
                    if os.path.exists(path):
                        continue

                    # 22 lines and 5 indents, classic
                    mask = font.getmask(char)
                    box = font.getbbox(char)
                    if mask.size == (0, 0) or box == badBox:
                        continue
                    im = Image.Image()._new(mask)

                    canvas = Image.new("L", (imageSize, imageSize), 0)
                    baseline = int(imageSize * 0.9)
                    offset = baseline - ascent
                    canvas.paste(im, ((imageSize - im.width) // 2, offset + box[1] - int(imageSize * 0.25)))

                    canvas.save(path)
            except Exception as e:
                print(fontPath, e)

            print(f"\rFonts serialized: {f + 1}/{len(fontPaths)}", end="")

        # Creating SDFs from all the bitmaps
        imagePaths = glob(os.path.join(config.directory, "bitmaps", "*"))
        for i, imagePath in enumerate(imagePaths):
            try:
                path = os.path.join(config.directory, "sdf", os.path.basename(imagePath).removesuffix(".bmp") + ".npy")
                if os.path.exists(path):
                    continue

                img = np.array(Image.open(imagePath))

                # Magic for sdf generation
                bits = img > (np.max(img) - np.min(img)) / 2
                sdf = dist(bits) - dist(~bits)

                np.save(path, sdf / imageSize)
            except Image.UnidentifiedImageError:
                print(imagePath, "Unidentified")

            print(f"\rSDFs generated: {i + 1}/{len(imagePaths)}", end="")

        # Lets us choose what kind of images to train on
        images = {}
        imagePaths = glob(os.path.join(config.directory, config.maps, "*"))
        for i, imagePath in enumerate(imagePaths):
            try:
                suffix = imagePath.split(".")[-1]
                if suffix == "npy":
                    images[os.path.basename(imagePath).removesuffix(".npy")] = np.load(imagePath)
                else:
                    images[os.path.basename(imagePath).removesuffix(".bmp")] = np.array(Image.open(imagePath)) / 255
            except Image.UnidentifiedImageError:
                print(imagePath, "Unidentified")
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

            names.append(key[:-2])

        names = np.array(names)
        pairs = np.array(pairs)
        mse = np.array(mse)
        letters = np.array(letters)

        plt.hist(mse)
        plt.grid()
        plt.show()

        mask = mse > np.percentile(mse, 40)

        # Manually excluding "too similar" pairs
        names = names[mask]
        pairs = pairs[mask]
        letters = letters[mask]

        self.names = names
        self.pairs = pairs
        self.letters = letters

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        lower, upper = self.pairs[item]

        lower = torch.tensor(lower, dtype=torch.float32)
        upper = torch.tensor(upper, dtype=torch.float32)

        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)

        return lower, upper, torch.tensor(characters.index(self.letters[item]), dtype=torch.long)
    
    @staticmethod
    def split(dataset, config):
        if config.split == "random":
            return torch.utils.data.random_split(dataset, [0.8, 0.2])
        
        # Designed to holdout a specific set of fonts for testing later
        if config.split == "holdout":
            testIndices = dataset.names.isin(config.standardFonts + config.stylizedFonts)
            trainIndices = ~testIndices
            return Subset(dataset, np.arange(len(dataset.names))[trainIndices]), Subset(dataset, np.arange(len(dataset.names))[testIndices])


if __name__ == "__main__":
    data = FontData(Config().load(os.path.join("configs", "config.json")).dataset)
    left, right = data[27]

    left = left.numpy()
    right = right.numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(left)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(right)
    plt.colorbar()

    plt.show()
