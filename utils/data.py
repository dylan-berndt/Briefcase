import torch
from torch.utils.data import Dataset, DataLoader, Subset
from .config import *
import numpy as np

import os
from glob import glob
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt as dist
from PIL import Image, ImageFont
from fontTools.ttLib import TTFont
import cv2
from concurrent.futures import ThreadPoolExecutor

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


# Lets us choose what kind of images to train on
def loadImage(imagePath):
    suffix = imagePath.split(".")[-1]
    if suffix == "npy":
        name, image = os.path.basename(imagePath).removesuffix(".npy"), np.load(imagePath)
    else:
        img = np.fromfile(imagePath, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        name, image = os.path.basename(imagePath).removesuffix(".bmp"), img

    return name, image


class FontData(Dataset):
    def __init__(self, config: Config, training=True):
        self.config = config
        self.method = self.config.method if "method" in self.config else "upper"

        if not os.path.exists(os.path.join(config.directory, "bitmaps")):
            os.mkdir(os.path.join(config.directory, "bitmaps"))

        if not os.path.exists(os.path.join(config.directory, "sdf")):
            os.mkdir(os.path.join(config.directory, "sdf"))

         # Just a bit of padding
        imageSize = int(config.fontSize * 1.5)
        ttfPaths = glob(os.path.join(config.directory, "fonts", "**", "*.ttf"), recursive=True)
        otfPaths = glob(os.path.join(config.directory, "fonts", "**", "*.otf"), recursive=True)
        fontPaths = ttfPaths + otfPaths

        self.fonts = {}
        self.fontMap = {}

        for f, fontPath in enumerate(fontPaths):
            if not os.path.isfile(fontPath):
                continue
            try:
                font = ImageFont.truetype(fontPath, config.fontSize)
                ascent, descent = font.getmetrics()
                # Gross way to do this, but I don't want another package
                standard = config.fontSize * (config.fontSize / ascent)
                font = ImageFont.truetype(fontPath, standard)
                ttf = TTFont(fontPath)
                badBox = font.getbbox('\uFFFF')

                fontName, fontStyle = font.getname()
                self.fonts[f"{fontName} {fontStyle}"] = font
                self.fontMap[f"{fontName} {fontStyle}"] = fontName

                imagePath = os.path.join(config.directory, "bitmaps", f"{fontName} {fontStyle} al.bmp")
                if os.path.exists(imagePath):
                    continue

                for char in characters:
                    case = "l" if char == char.lower() else "u"
                    name = f"{fontName} {fontStyle} {char.lower()}{case}"
                    path = os.path.join(config.directory, "bitmaps", name + ".bmp")

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

                    canvas.save(path)
            except Exception as e:
                print(fontPath, e)

            print(f"\rFonts serialized: {f + 1}/{len(fontPaths)}", end="")

        # Creating SDFs from all the bitmaps
        if config.maps == "sdf":
            imagePaths = glob(os.path.join(config.directory, "bitmaps", "*"))
            for i, imagePath in enumerate(imagePaths):
                try:
                    path = os.path.join(config.directory, "sdf", os.path.basename(imagePath).removesuffix(".bmp") + ".npy")
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
        imagePaths = glob(os.path.join(config.directory, config.maps, "*"))
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

        # plt.hist(mse, bins=10)
        # plt.grid()
        # plt.show()

        if training:
            mask = mse > np.percentile(mse, 40)

            # Manually excluding "too similar" pairs
            names = names[mask]
            pairs = pairs[mask]
            letters = letters[mask]

        self.names = names
        self.pairs = pairs
        self.letters = letters

        self.index = np.arange(len(self.pairs))

        print()

    def mask(self, mask):
        pass

    def __len__(self):
        if self.method == "masked":
            return len(self.index) * 2

        return len(self.index)
    
    def getPair(self, i):
        item = self.index[i]
        lower, upper = self.pairs[item]

        lower = torch.tensor(lower, dtype=torch.float32)
        upper = torch.tensor(upper, dtype=torch.float32)

        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)

        character = torch.tensor(characters.index(self.letters[item]), dtype=torch.long)

        if self.method == "lower":
            return {"inputs": upper, "outputs": lower, "class": character,
                    "name": self.names[item], "letter": self.letters[item]}
        
        return {"inputs": lower, "outputs": upper, "class": character,
                "name": self.names[item], "letter": self.letters[item]}
    
    def getMasked(self, i):
        item = self.index[i // 2]
        lower, upper = self.pairs[item]
        image = lower if (i % 2 == 0) else upper
        maskedImage = image.copy()

        patchSize = maskedImage.shape[0] // 4
        x = np.random.randint(0, maskedImage.shape[0] - patchSize + 1)
        y = np.random.randint(0, maskedImage.shape[1] - patchSize + 1)


        maskedImage[x: x + patchSize, y: y + patchSize] = 0

        noiseMask = np.random.randint(0, 100, size=maskedImage.shape) < 40
        maskedImage[noiseMask] = 0

        maskedImage = torch.tensor(maskedImage, dtype=torch.float32).unsqueeze(-1)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(-1)

        letter = self.letters[item] if (i % 2 == 0) else self.letters[item].upper()
        character = torch.tensor(characters.index(letter), dtype=torch.long)

        return {"inputs": maskedImage, "outputs": image, "class": character,
                "name": self.names[item], "letter": letter}
    
    @staticmethod
    def collate(samples):
        inputs = torch.stack([sample["inputs"] for sample in samples], dim=0)
        outputs = torch.stack([sample["outputs"] for sample in samples], dim=0)
        characters = torch.stack([sample["class"] for sample in samples], dim=0)

        return inputs, outputs, characters

    def __getitem__(self, item):
        if self.method == "masked":
            return self.getMasked(item)
        
        return self.getPair(item)
    
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
