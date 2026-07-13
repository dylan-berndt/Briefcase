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

from pygtrie import CharTrie


GENERIC_FONTS = {"noto", "unifont", "quivira", "symbola", "dejavu", "gnu unifont"}


def loadDescriptionsFromSource(config: Config):
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

    return descriptions


class CombinedQueryData:
    def __init__(self, config, training=False, tokenizer="bert-base-uncased", limit=None, multiClass=False):
        self.setTokenizer(tokenizer)
        self.config = config
        self.multiClass = multiClass

        imageSize = int(config.fontSize * 1.5)

        self.transforms = v2.Compose([
            v2.RandomResizedCrop(size=(imageSize, imageSize), scale=(0.7, 1.0), ratio=(0.75, 1.333)),
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

        if not os.path.exists(os.path.join("results", "fontQueries.json")) or multiClass:
            self.queries = False
            self.descriptions = loadDescriptionsFromSource(config)
        else:
            with open(os.path.join("results", "fontQueries.json"), "r") as file:
                print("USING GENERATED QUERIES")
                self.queries = True
                self.descriptions = json.load(file)
                self.descriptions = {
                    k: v for k, v in self.descriptions.items()
                    if not any(k.lower().startswith(prefix) for prefix in GENERIC_FONTS)
                }

        print(len(self.names), len(self.descriptions), len(self.paths))

        trie = CharTrie()
        for key in self.descriptions:
            trie[key] = key

        solidNames = []
        viable = []
        for i, name in enumerate(self.names):
            # longest_prefix is O(len(name)) instead of O(num_keys)
            match = trie.longest_prefix(name)
            if match.key is not None:
                solidNames.append(match.key)
                viable.append(True)
            else:
                solidNames.append(None)
                viable.append(False)

        self.solidNames = np.array(solidNames)
                
        if training:
            viable = np.array(viable, dtype=bool)
            print(f"{np.mean(viable) * 100:.2f}% of fonts have descriptions")
            self.index = np.arange(len(self.paths))[viable]
        else:
            self.index = np.arange(len(self.paths))

        if multiClass:
            counts = {}
            for description in self.descriptions.values():
                for tag in description.adjectives + list(description.tags.keys()):
                    counts[tag] = counts.get(tag, 0) + 1

            self.vocab = sorted(tag for tag, count in counts.items() if count >= 10)
            
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

        name = self.solidNames[imageIndex]

        leftImage = self._jiggle(torch.tensor(image, dtype=torch.float32))
        # leftImage = torch.tensor(image, dtype=torch.float32).unsqueeze(-1)

        letter = self.letters[imageIndex] if (i % 2 == 0) else self.letters[imageIndex].upper()
        # Bastard: "ԵՒ" 
        if letter in characters:
            num = characters.index(letter)
        else:
            num = -1
        character = torch.tensor(num, dtype=torch.long)

        fontName = self.fontMap[name]

        if not self.multiClass:
            if not self.queries:
                description = self.descriptions[fontName].sample()
            else:
                description = random.choice(self.descriptions[fontName])
        else:
            desc = self.descriptions[fontName]
            tags = set(desc.adjectives + list(desc.tags.keys()))
            description = [1 if item in tags else 0 for item in self.vocab]
            description = torch.tensor(description, dtype=torch.long)

        return {"inputs": leftImage, "fontID": self.fontNum[name],
                "class": character, "description": description}
    
    @staticmethod
    def collate(samples):
        inputs = torch.stack([sample["inputs"] for sample in samples], dim=0)
        characters = torch.stack([sample["class"] for sample in samples], dim=0)
        names = torch.tensor([sample["fontID"] for sample in samples], dtype=torch.long)

        if isinstance(samples[0]["description"], torch.Tensor):
            tokens = torch.stack([sample["description"] for sample in samples], dim=0)
        else:
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
        trainIndexMask = np.isin(dataset.solidNames[dataset.index], trainIDs)

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
        self.model: CLIPTextModel = CLIPTextModel.from_pretrained(modelName)
        # self.model.requires_grad_(False)
        cfg = AutoConfig.from_pretrained(modelName)
        sourceDim = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.projection_dim
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(sourceDim, sharedDim),
            nn.LayerNorm(sharedDim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(sharedDim, sharedDim),
            nn.LayerNorm(sharedDim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(sharedDim, sharedDim)
        )

    def forward(self, text):
        features = self.model(**text).pooler_output
        return self.head(features)
    
    @staticmethod
    def load(path, name="checkpoint"):
        modelPath = os.path.join(path, f"{name}.pt")
        configPath = os.path.join(path, "config.json")

        loadedConfig = Config().load(configPath)
        loadedModel = CLIPTextEmbedder(loadedConfig.model)

        loaded = torch.load(modelPath, weights_only=False, map_location="cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(loaded, "state_dict"):
            loaded = loaded.state_dict()
        loadedModel.load_state_dict(loaded)
        loadedModel.eval()

        return loadedModel, loadedConfig


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

