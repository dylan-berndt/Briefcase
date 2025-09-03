import torch
from torch.utils.data import Dataset, DataLoader
from config import *
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
characters = [chr(c) for c in characters]
characters = characters + [c.upper() for c in characters]
print(characters)

# 2 characters technically, and messes with checks for empty glyphs
characters.remove("ԵՒ")


class FontData(Dataset):
    def __init__(self, config: Config):
        for fontPath in glob(os.path.join("data", "fonts", "*")):
            try:
                font = ImageFont.truetype(fontPath, config.fontSize)
                badBox = font.getbbox('\uFFFF')
                for char in characters:
                    case = "l" if char == char.lower() else "u"
                    name = os.path.basename(fontPath) + " " + char.lower() + case
                    path = os.path.join("data", "bitmaps", name + ".bmp")
                    if os.path.exists(path):
                        continue

                    # 22 lines and 5 indents
                    mask = font.getmask(char)
                    box = font.getbbox(char)
                    if mask.size == (0, 0) or box == badBox:
                        continue
                    im = Image.Image()._new(mask)

                    # Resize bc fonts are evil dastardly things
                    maxDim = max(im.width, im.height)
                    w = int(im.width * (config.fontSize / maxDim))
                    h = int(im.height * (config.fontSize / maxDim))
                    im = im.resize((w, h))

                    im.save(path)
            except Exception as e:
                print(fontPath, e)

        # Just a bit of padding
        imageSize = int(config.fontSize * 1.25)
        # Creating SDFs from all the bitmaps
        for imagePath in glob(os.path.join("data", "bitmaps", "*")):
            path = os.path.join("data", "sdf", os.path.basename(imagePath).removesuffix(".bmp") + ".npy")
            if os.path.exists(path):
                continue

            img = np.array(Image.open(imagePath))
            array = np.zeros([imageSize, imageSize])

            # Place in center of image for standardization purposes
            offset = (imageSize - img.shape[0]) // 2, (imageSize - img.shape[1]) // 2
            array[offset[0]:offset[0] + img.shape[0], offset[1]:offset[1] + img.shape[1]] = img

            # Magic for sdf generation
            bits = array > (np.max(array) - np.min(array)) / 2
            sdf = dist(bits) - dist(~bits)

            np.save(path, sdf / imageSize)

        images = {}
        for sdfPath in glob(os.path.join("data", "sdf", "*")):
            images[os.path.basename(sdfPath).removesuffix(".npy")] = np.load(sdfPath)

        pairs = []
        mse = []
        for key in images:
            case = key[-1]
            if case == "l":
                continue
            other = key[:-1] + "l"

            if other not in images:
                continue

            mse.append(np.mean(np.power(images[other] - images[key], 2)))

            pairs.append((images[other], images[key]))

        pairs = np.array(pairs)
        mse = np.array(mse)

        # Manually excluding "too similar" pairs
        pairs = pairs[mse > 0.005]

        # plt.hist(mse)
        # plt.grid()
        # plt.show()

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        lower, upper = self.pairs[item]

        lower = torch.tensor(lower, dtype=torch.float32)
        upper = torch.tensor(upper, dtype=torch.float32)

        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)

        return lower, upper


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
