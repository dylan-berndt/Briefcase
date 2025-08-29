import torch
from torch.utils.data import Dataset, DataLoader
from config import *
import numpy as np

import os
from glob import glob
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt as dist
from PIL import Image, ImageFont

characters = [chr(c) for c in list(range(ord('A'), ord('A') + 26)) + list(range(ord('a'), ord('a') + 26))]


class FontData(Dataset):
    def __init__(self, config: Config):
        for fontPath in glob(os.path.join("data", "fonts", "*")):
            try:
                font = ImageFont.truetype(fontPath, config.fontSize)
                for char in characters:
                    # 22 lines and 5 indents
                    im = Image.Image()._new(font.getmask(char))

                    # Resize bc fonts are evil dastardly things
                    maxDim = max(im.width, im.height)
                    w = int(im.width * (config.fontSize / maxDim))
                    h = int(im.height * (config.fontSize / maxDim))
                    im = im.resize((w, h))

                    case = "l" if char == char.lower() else "u"
                    path = os.path.join("data", "bitmaps", os.path.basename(fontPath) + " " + char.lower() + case + ".bmp")
                    im.save(path)
            except Exception as e:
                print(fontPath, e)

        # Just a bit of padding
        imageSize = int(config.fontSize * 1.25)
        for imagePath in glob(os.path.join("data", "bitmaps", "*")):
            img = np.array(Image.open(imagePath))
            array = np.zeros([imageSize, imageSize])

            # Place in center of image for standardization purposes
            offset = (imageSize - img.shape[0]) // 2, (imageSize - img.shape[1]) // 2
            array[offset[0]:offset[0] + img.shape[0], offset[1]:offset[1] + img.shape[1]] = img

            # Magic for sdf generation
            bits = array > (np.max(array) - np.min(array)) / 2
            sdf = dist(bits) - dist(~bits)

            path = os.path.join("data", "sdf", os.path.basename(imagePath).removesuffix(".bmp") + ".npy")
            np.save(path, sdf / imageSize)

        images = {}
        for sdfPath in glob(os.path.join("data", "sdf", "*")):
            images[os.path.basename(sdfPath).removesuffix(".npy")] = np.load(sdfPath)

        pairs = []
        for key in images:
            case = key[-1]
            if case == "l":
                continue
            other = key[:-1] + "l"

            pairs.append((images[other], images[key]))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        left, right = self.pairs[item]

        left = torch.tensor(left, dtype=torch.float32)
        right = torch.tensor(right, dtype=torch.float32)

        return left, right


if __name__ == "__main__":
    data = FontData(Config().load(os.path.join("configs", "config.json")).dataset)
    left, right = data[0]

    left = left.numpy()
    right = right.numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(left)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(right)
    plt.colorbar()

    plt.show()
