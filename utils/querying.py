# Utility script to preprocess fonts and descriptions from the Google Fonts repository
# Needs to include name, tags, description
# Then filter description for adjectives and adjective-noun pairs with spacy
# When loading later, can shuffle name, tags, description and randomly include / exclude for robustness

import os

from .unet import *
from .pretraining import *
from .loaders import *
from .vit import *
import cv2
import random
import pandas as pd
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel, CLIPImageProcessor, BertModel, AutoConfig, AutoModel
from collections import defaultdict

from multiprocessing import Pool

from bs4 import BeautifulSoup
import spacy

from torchvision.transforms import v2

import math


# This is very specifically tailored to the Google Fonts repository
class QueryData(FontData):
    def __init__(self, config, training=False, tokenizer="bert-base-uncased"):
        super().__init__(config, training)

        self.setTokenizer(tokenizer)

        self.descriptions = loadGoogleDescriptions(config.directory)

        names = np.array([self.fontMap[name] for name in self.names])
        viable = np.isin(names, list(self.descriptions.keys()))
        print(f"{np.mean(viable) * 100:.2f}% of fonts have descriptions")
        self.index = np.arange(len(self.pairs))[viable]
        self.fontNum = {name: i for i, name in enumerate(self.fontMap.keys())}

    def setTokenizer(self, name):
        Description.tokenizer = AutoTokenizer.from_pretrained(name)

    def __getitem__(self, i):
        data = super().__getitem__(i)

        fontName = self.fontMap[data["name"]]
        description = self.descriptions[fontName].sample()

        data["description"] = description
        data["fontID"] = self.fontNum[data["name"]]

        return data
    
    @staticmethod
    def collate(samples):
        inputs = torch.stack([sample["inputs"] for sample in samples], dim=0)
        outputs = torch.stack([sample["outputs"] for sample in samples], dim=0)
        characters = torch.stack([sample["class"] for sample in samples], dim=0)
        names = torch.tensor([sample["fontID"] for sample in samples], dtype=torch.long)

        tokens = Description.tokenizer([sample["description"] for sample in samples],
                                       padding="longest", truncation=True,
                                       return_tensors="pt")

        return inputs, outputs, names, tokens, characters
    
    @staticmethod
    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234, batchSize=128):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        fontIDs = list(dataset.fonts.keys())
        trainIDs = np.array(fontIDs)[np.random.choice(len(fontIDs), int(len(fontIDs) * trainSplit), replace=False)]
        trainIndexMask = np.isin(dataset.names[dataset.index], trainIDs)

        # Pretty sure this flattens correctly
        if len(trainIndexMask) != len(dataset):
            trainIndexMask = np.stack([trainIndexMask, trainIndexMask], axis=1).flatten()

        trainIndex = np.arange(len(dataset))[trainIndexMask]
        testIndex = np.arange(len(dataset))[~trainIndexMask]

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        train = DataLoader(train, batch_size=batchSize, collate_fn=dataset.collate, shuffle=shuffle)
        test = DataLoader(test, batch_size=batchSize, collate_fn=dataset.collate, shuffle=shuffle)

        return train, test


# Designed for the MyFonts dataset from Rochester. Does not include actual font files :(
class MyFontsQueryData(QueryData):
    def __init__(self, config, training=False, tokenizer="bert-base-uncased", limit=None):
        self.setTokenizer(tokenizer)

        self.config = config
        self.method = self.config.method if "method" in self.config else "upper"

        if not os.path.exists(os.path.join(config.directory, "smallimage")):
            os.mkdir(os.path.join(config.directory, "smallimage"))

        if len(glob(os.path.join(config.directory, "smallimage", "*.bmp"))) == 0:
            imagePaths = glob(os.path.join(config.directory, "fontimage", "*.png"))
            tasks = [(path, config.fontSize) for path in imagePaths]
            with Pool(processes=2) as pool:
                for i, (name, array) in enumerate(pool.imap(loadRochesterImage, tasks, chunksize=1000)):
                    if name == None:
                        continue

                    img = Image.fromarray((array * 255).astype(np.uint8)).convert('L')
                    fontName = name.split("_")[0]
                    letter = name[-1].lower()
                    case = "u" if name[-1] == name[-2] else "l"
                    imageName = f"{fontName} {letter}{case}.bmp"
                    img.save(os.path.join(config.directory, "smallimage", imageName))
                    if i % 100 == 0:
                        print(f"\rImages converted: {i + 1}/{len(imagePaths)}", end="")

        print()

        images = {}
        imagePaths = glob(os.path.join(config.directory, "smallimage", "*.bmp"))
        if limit is not None:
            imagePaths = imagePaths[:limit]
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            futures = {executor.submit(loadImage, p): p for p in imagePaths}
            for i, future in enumerate(as_completed(futures)):
                results = future.result()
                name, image = results
                images[name] = image

                if i % 100 == 0:
                    print(f"\rImages loaded: {i + 1}/{len(imagePaths)}", end="")

        print()

        self.descriptions = {}
        descriptionPaths = glob(os.path.join(config.directory, "taglabel", "*"))
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            futures = {executor.submit(loadRochesterDescription, p): p for p in descriptionPaths}
            for i, future in enumerate(as_completed(futures)):
                results = future.result()
                name, description = results
                self.descriptions[name] = description

                if i % 100 == 0:
                    print(f"\rDescriptions loaded: {i + 1}/{len(descriptionPaths)}", end="")

        print()

        pairs = []
        letters = []
        names = []
        mse = []
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

        self.names = np.array(names)
        self.pairs = np.array(pairs)
        self.letters = np.array(letters)

        mse = np.array(mse)

        if training:
            mask = mse > np.percentile(mse, 40)

            # Manually excluding "too similar" pairs
            self.names = self.names[mask]
            self.pairs = self.pairs[mask]
            self.letters = self.letters[mask]

        viable = np.isin(self.names, list(self.descriptions.keys()))
        print(f"{np.mean(viable) * 100:.2f}% of fonts have descriptions")
        self.index = np.arange(len(self.pairs))[viable]

        self.fontMap = {key: key for key in self.descriptions}
        self.fontNum = {key: i for i, key in enumerate(self.descriptions)}
        self.fonts = {key: None for key in self.descriptions}

        # self.fonts Name -> Loaded Font 


class CombinedQueryData:
    def __init__(self, config, training=False, tokenizer="bert-base-uncased", limit=None):
        self.setTokenizer(tokenizer)
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

        if "directories" in config:
            descriptions = {}
            for directoryType in config.directories:
                if directoryType == "myFonts":
                    descriptions.update(loadRochesterDescriptions(config.directories.myFonts))
                if directoryType == "standard":
                    for directory in config.directories.standard:
                        if directory == "google":
                            descriptions.update(loadGoogleDescriptions(directory))
                        if directory == "dafont":
                            descriptions.update(loadDaFontDescriptions(directory))
        else:
            descriptions = loadRochesterDescriptions(config.directory)

        self.descriptions = descriptions

        viable = np.isin(self.names, list(self.descriptions.keys()))
        print(f"{np.mean(viable) * 100:.2f}% of fonts have descriptions")
        self.index = np.arange(len(self.paths))[viable]

        self.fontMap = {key: key for key in self.descriptions}
        self.fontNum = {key: i for i, key in enumerate(self.descriptions)}
        self.fonts = {key: None for key in self.descriptions}

    def setTokenizer(self, name):
        Description.tokenizer = AutoTokenizer.from_pretrained(name)

    def __len__(self):
        return len(self.index)
    
    def _jiggle(self, image):
        image = image.unsqueeze(-1).permute(2, 0, 1)
        image = self.transforms(image)
        return image.permute(1, 2, 0)
    
    def __getitem__(self, i):
        imageIndex = self.index[i]
        imagePath = self.paths[imageIndex]
        _, image = loadImage(imagePath)

        if image is None:
            imageSize = int(self.config.fontSize * 1.5)
            image = np.zeros((imageSize, imageSize), dtype=np.float32)

        name = self.names[imageIndex]

        leftImage = self._jiggle(torch.tensor(image, dtype=torch.float32))

        letter = self.letters[imageIndex] if (i % 2 == 0) else self.letters[imageIndex].upper()
        # Bastard: "ԵՒ" 
        if letter in characters:
            num = characters.index(letter)
        else:
            num = -1
        character = torch.tensor(num, dtype=torch.long)

        fontName = self.fontMap[name]
        description = self.descriptions[fontName].sample()

        return {"inputs": leftImage, "fontID": self.fontNum[name],
                "class": character, "description": description}
    
    @staticmethod
    def collate(samples):
        inputs = torch.stack([sample["inputs"] for sample in samples], dim=0)
        characters = torch.stack([sample["class"] for sample in samples], dim=0)
        names = torch.tensor([sample["fontID"] for sample in samples], dtype=torch.long)

        tokens = Description.tokenizer([sample["description"] for sample in samples],
                                       padding="longest", truncation=True,
                                       return_tensors="pt")

        return inputs, names, tokens, characters
    
    @staticmethod
    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234, batchSize=128):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        fontIDs = list(dataset.fonts.keys())
        trainIDs = np.array(fontIDs)[np.random.choice(len(fontIDs), int(len(fontIDs) * trainSplit), replace=False)]
        trainIndexMask = np.isin(dataset.names[dataset.index], trainIDs)

        # Pretty sure this flattens correctly
        if len(trainIndexMask) != len(dataset):
            trainIndexMask = np.stack([trainIndexMask, trainIndexMask], axis=1).flatten()

        trainIndex = np.arange(len(dataset))[trainIndexMask]
        testIndex = np.arange(len(dataset))[~trainIndexMask]

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        train = DataLoader(train, batch_size=batchSize, collate_fn=dataset.collate, shuffle=shuffle)
        test = DataLoader(test, batch_size=batchSize, collate_fn=dataset.collate, shuffle=shuffle)

        return train, test

    
class CLIPEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LazyLinear(config.sharedDim),
            nn.ReLU(),
            nn.LazyLinear(config.sharedDim)
        )

    def preprocess(self, x):
        x = x * 255.0
        x = x.clamp(0, 255)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x / 255.0 - mean) / std

        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return x

    def forward(self, x):
        with torch.no_grad():
            x = x.permute(0, 3, 1, 2)
            x = x.repeat(1, 3, 1, 1)
            x = self.preprocess(x)
            x = self.model(x)
        if self.outputDimension is not None:
            return self.head(x.pooler_output)
        return x

    def activations(self, x):
        outputs = self.forward(x)
        return [outputs.pooler_output]
    

class CLIPTextEmbedder(nn.Module):
    def __init__(self, modelName, sharedDim):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(modelName)
        self.model.requires_grad_(False)
        cfg = AutoConfig.from_pretrained(modelName)
        sourceDim = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.projection_dim
        self.head = nn.Sequential(
            nn.Linear(sourceDim, sharedDim),
            nn.ReLU(),
            nn.Linear(sharedDim, sharedDim)
        )

    def forward(self, text):
        with torch.no_grad():
            features = self.model(**text).pooler_output
        return self.head(features)


class BERTTextEmbedder(nn.Module):
    def __init__(self, modelName, sharedDim):
        super().__init__()
        self.model = BertModel.from_pretrained(modelName)
        self.model.requires_grad_(False)
        cfg = AutoConfig.from_pretrained(modelName)
        sourceDim = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.projection_dim
        self.head = nn.Sequential(
            nn.Linear(sourceDim, sharedDim),
            nn.ReLU(),
            nn.Linear(sharedDim, sharedDim)
        )

    def forward(self, text):
        with torch.no_grad():
            features = self.model(**text).pooler_output
        return self.head(features)

