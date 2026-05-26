import torch
from torch.utils.data import Dataset, DataLoader, Subset
from .config import *
import numpy as np

import os
import matplotlib.pyplot as plt

from .loaders import *

from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"


class FontData(Dataset):
    def __init__(self, config: Config, training=True):
        self.config = config
        self.method = self.config.method if "method" in self.config else "upper"

        data = loadFontSet(config.directory, config.fontSize, config.maps)

        names, letters, pairs, mse = data["names"], data["letters"], data["pairs"], data["mse"]

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

        self.fontNum = {key: i for i, key in enumerate(list(set(self.names.tolist())))}

        print()

    def mask(self, mask):
        pass

    def __len__(self):
        if self.method == "masked" or self.method == "none":
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
                    "name": self.names[item], "letter": self.letters[item], "id": self.fontNum[self.names[item]]}
        
        return {"inputs": lower, "outputs": upper, "class": character,
                "name": self.names[item], "letter": self.letters[item], "id": self.fontNum[self.names[item]]}
    
    def getMasked(self, i):
        item = self.index[i // 2]
        lower, upper = self.pairs[item]
        image = lower if (i % 2 == 0) else upper

        if self.method == "masked":
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

        if self.method == "none":
            return {"inputs": image, "outputs": image, "class": character,
                    "name": self.names[item], "letter": letter, "id": self.fontNum[self.names[item]]}

        return {"inputs": maskedImage, "outputs": image, "class": character,
                "name": self.names[item], "letter": letter, "id": self.fontNum[self.names[item]]}
    
    @staticmethod
    def collate(samples):
        inputs = torch.stack([sample["inputs"] for sample in samples], dim=0)
        outputs = torch.stack([sample["outputs"] for sample in samples], dim=0)
        characters = torch.stack([sample["class"] for sample in samples], dim=0)
        ids = torch.tensor([sample["id"] for sample in samples], dtype=torch.int32)

        return inputs, outputs, characters, ids

    def __getitem__(self, item):
        if self.method == "masked" or self.method == "none":
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
        

class PairedImageData(FontData):
    def __init__(self, config, training=False, limit=None):
        self.config = config

        imageSize = int(config.fontSize * 1.5)

        self.transforms = v2.Compose([
            v2.RandomResizedCrop(size=(imageSize, imageSize), scale=(0.7, 1.0), ratio=(1.0, 1.0)),
            v2.RandomRotation(degrees=25)
        ])

        if "directories" in config:
            names = []
            letters = []
            paths = []
            if "myFonts" in config.directories:
                data = loadMyFontsImagePaths(config.directories.myFonts, config.fontSize)
                names.append(data["names"]); letters.append(data["letters"]); paths.append(data["paths"])
            if "standard" in config.directories:
                for directory in config.directories.standard:
                    data = collectFontSetPaths(directory, config.fontSize, config.maps)
                    names.append(data["names"]); letters.append(data["letters"]); paths.append(data["paths"])

            names = np.concatenate(names, axis=0); letters = np.concatenate(letters, axis=0); paths = np.concatenate(paths, axis=0)
        else:
            data = loadMyFontsImagePaths(config.directory, config.fontSize)
            names, letters, paths = data["names"], data["letters"], data["paths"]

        self.names = names
        self.letters = letters
        self.paths = paths

        indices = np.argsort(self.names)

        self.names = self.names[indices]
        self.letters = self.letters[indices]
        self.paths = self.paths[indices]

        # fonts, glyphsPerFont = np.unique(self.names, return_counts=True)
        # interactions = np.power(glyphsPerFont, 2)

        fonts, glyphsPerFont = np.unique(self.names, return_counts=True)

        self.fonts = fonts
        self.glyphsPerFont = glyphsPerFont

        # starting glyph index for each font
        self.fontOffsets = np.concatenate([[0], np.cumsum(glyphsPerFont[:-1])])

        # number of pair interactions per font
        self.fontPairCounts = glyphsPerFont ** 2

        # starting pair index for each font
        self.fontPairOffsets = np.concatenate([[0], np.cumsum(self.fontPairCounts[:-1])])

        self.totalPairs = int(np.sum(self.fontPairCounts))

        self.fontNum = {key: i for i, key in enumerate(list(set(self.names.tolist())))}

    def decodePairIndex(self, idx):
        # which font block this pair belongs to
        fontIdx = np.searchsorted(
            self.fontPairOffsets,
            idx,
            side="right"
        ) - 1

        localPairIdx = idx - self.fontPairOffsets[fontIdx]

        glyphCount = self.glyphsPerFont[fontIdx]

        leftLocal = localPairIdx // glyphCount
        rightLocal = localPairIdx % glyphCount

        base = self.fontOffsets[fontIdx]

        leftIdx = base + leftLocal
        rightIdx = base + rightLocal

        return leftIdx, rightIdx

    def __len__(self):
        return len(self.index)
    
    def _jiggle(self, image):
        image = image.unsqueeze(-1).permute(2, 0, 1)
        image = self.transforms(image)
        return image.permute(1, 2, 0)
    
    def __getitem__(self, i):
        leftIndex, rightIndex = self.decodePairIndex(i)

        # leftIndex = self.leftIndex[i]
        # rightIndex = self.rightIndex[i]

        leftImagePath = self.paths[leftIndex]
        rightImagePath = self.paths[rightIndex]

        _, leftImage = loadImage(leftImagePath)
        _, rightImage = loadImage(rightImagePath)

        if leftImage is None:
            imageSize = int(self.config.fontSize * 1.5)
            leftImage = np.zeros((imageSize, imageSize), dtype=np.float32)

        if rightImage is None:
            imageSize = int(self.config.fontSize * 1.5)
            rightImage = np.zeros((imageSize, imageSize), dtype=np.float32)

        name = self.names[leftIndex]

        leftImage = self._jiggle(torch.tensor(leftImage, dtype=torch.float32))
        rightImage = self._jiggle(torch.tensor(rightImage, dtype=torch.float32))

        letter = self.letters[leftIndex] if (i % 2 == 0) else self.letters[leftIndex].upper()
        # Bastard: "ԵՒ" 
        if letter in characters:
            num = characters.index(letter)
        else:
            num = -1
        character = torch.tensor(num, dtype=torch.long)

        return {"inputs": leftImage, "outputs": rightImage, "name": name,
                "class": character, "letter": letter, "id": self.fontNum[name]}
    
    # @staticmethod
    # def split(dataset, config):
    #     fonts = np.unique(dataset.names)
    #     np.random.shuffle(fonts)
    #     trainFonts = set(fonts[:int(len(fonts) * 0.8)])
        
    #     trainMask = np.isin(dataset.names, list(trainFonts))
    #     testMask = ~trainMask
        
    #     trainIndices = np.where(trainMask[dataset.leftIndex] & trainMask[dataset.rightIndex])[0]
    #     testIndices = np.where(testMask[dataset.leftIndex] & testMask[dataset.rightIndex])[0]
        
    #     return torch.utils.data.Subset(dataset, trainIndices), torch.utils.data.Subset(dataset, testIndices)

    @staticmethod
    def split(dataset, config):
        fonts = np.array(dataset.fonts)

        np.random.shuffle(fonts)

        split = int(len(fonts) * 0.8)

        trainFonts = set(fonts[:split])

        trainPairIndices = []
        testPairIndices = []

        for i, font in enumerate(dataset.fonts):
            start = dataset.fontPairOffsets[i]
            count = dataset.fontPairCounts[i]

            indices = np.arange(start, start + count)

            if font in trainFonts:
                trainPairIndices.append(indices)
            else:
                testPairIndices.append(indices)

        trainPairIndices = np.concatenate(trainPairIndices)
        testPairIndices = np.concatenate(testPairIndices)

        return (
            torch.utils.data.Subset(dataset, trainPairIndices),
            torch.utils.data.Subset(dataset, testPairIndices)
        )
        

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
