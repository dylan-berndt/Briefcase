# Utility script to preprocess fonts and descriptions from the Google Fonts repository
# Needs to include name, tags, description
# Then filter description for adjectives and adjective-noun pairs with spacy
# When loading later, can shuffle name, tags, description and randomly include / exclude for robustness

import os

from .model import *
from .data import *
import cv2
import random
import pandas as pd
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel, CLIPImageProcessor, BertModel, AutoConfig
from collections import defaultdict

from multiprocessing import Pool

from bs4 import BeautifulSoup
import spacy

nlp = spacy.load("en_core_web_sm")


def getAdjectives(filePath):
    with open(filePath, "r", encoding="utf-8") as file:
        data = file.read()
        soup = BeautifulSoup(data, 'html.parser')
        text = soup.get_text()

        last = None
        adjectives = []

        # Yippee for state machines
        for token in nlp(text):
            if token.pos_ == "ADJ":
                last = token.text
            elif token.pos_ == "NN":
                if last is not None:
                    adjectives.append(f"{last} {token.text}")
                    last = None
            elif last is not None:
                adjectives.append(last)
                last = None
        
        return adjectives
    

class Description:
    tokenizer = None
    maxDescriptors = 7

    def __init__(self, name, adjectives, tags=None):
        tags = tags if tags is not None else {}
        
        self.name = name
        self.adjectives = adjectives
        self.tags = tags

    def sample(self):
        if self.tags is not None:
            tags = [tag for tag, value in self.tags.items() if random.uniform(0, 1) < value]

            descriptors = self.adjectives + tags
        else:
            descriptors = self.adjectives
        numDescriptors = int(random.uniform(0.6, 1.0) * Description.maxDescriptors)
        chosenDescriptors = random.sample(descriptors, min(len(descriptors), numDescriptors))

        joined = ", ".join(chosenDescriptors) + " font"
        if random.uniform(0, 1) > 0.2:
            joined = joined + " named " + self.name

        return "a " + joined

    def __len__(self):
        descriptors = self.adjectives + list(self.tags.keys())
        description = ", ".join(descriptors) + " font named " + self.name
        tokens = Description.tokenizer([description], padding=False, return_tensors="pt")
        return len(tokens["input_ids"][0])
    

def loadFolderDescription(walk):
    root, dirs, files = walk
    if not any([file.endswith("METADATA.pb") for file in files]):
        return None
    
    fontNames = []
    fontStyles = []
    caption = None
    
    for file in glob(os.path.join(root, "**", "*.html"), recursive=True):
        if file.endswith(".html"):
            caption = getAdjectives(file)

    otf = glob(os.path.join(root, "**", "*.otf"), recursive=True)
    ttf = glob(os.path.join(root, "**", "*.ttf"), recursive=True)
    for filePath in (otf + ttf):
        try:
            font = ImageFont.truetype(filePath, 32)
            fontName, fontStyle = font.getname()
            fontNames.append(fontName)
            fontStyles.append(fontStyle)

        except Exception as e:
            print(filePath, e)
            continue

    return fontNames, fontStyles, caption


# This is very specifically tailored to the Google Fonts repository
class QueryData(FontData):
    def __init__(self, config, training=False, tokenizer="bert-base-uncased"):
        super().__init__(config, training)

        self.setTokenizer(tokenizer)

        self.descriptions = {}

        walks = os.walk(os.path.join(config.directory, "fonts"))
        roots = [root for root, dirs, files in walks]
        walks = os.walk(os.path.join(config.directory, "fonts"))
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            for i, result in enumerate(executor.map(loadFolderDescription, walks)):
                if result is None:
                    continue

                fontNames, fontStyles, caption = result

                if caption is None:
                    print(roots[i], fontNames)
                    continue

                for j in range(len(fontNames)):
                    fontName = fontNames[j]
                    fontStyle = fontStyles[j]
                    self.descriptions[fontName] = Description(f"{fontName} {fontStyle}", caption, {fontStyle: 0.2})

                if i % 100 == 0:
                    print(f"\rPaths checked: {i + 1}/{len(roots)}", end="")

        print()
                    
        tagFile = os.path.join(config.directory, "fonts", "tags", "all", "families.csv")
        tagDF = pd.read_csv(tagFile, names=["family", "na", "tags", "weight"])
        for family in tagDF.family.unique():
            familyDF = tagDF[tagDF.family == family]
            tags = familyDF.tags.tolist()
            weights = familyDF.weight.tolist()

            tagDict = {}
            for i, tagSet in enumerate(tags):
                tag = tagSet.split("/")[-1]
                tagDict[tag] = weights[i] / 100

            if family not in self.descriptions:
                continue

            self.descriptions[family].tags = tagDict

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
        # characters = torch.stack([sample["class"] for sample in samples], dim=0)
        names = torch.tensor([sample["fontID"] for sample in samples], dtype=torch.long)

        tokens = Description.tokenizer([sample["description"] for sample in samples],
                                       padding="longest", truncation=True,
                                       return_tensors="pt")

        return inputs, outputs, names, tokens
    
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

        train = DataLoader(train, batch_size=batchSize, collate_fn=dataset.collate,
                           generator=torch.Generator(device), shuffle=shuffle)
        test = DataLoader(test, batch_size=batchSize, collate_fn=dataset.collate,
                          generator=torch.Generator(device), shuffle=shuffle)

        return train, test


class Loader:
    def __init__(self, fontSize):
        self.fontSize = fontSize

    def loadRochesterImage(self, imagePath):
        image = Image.open(imagePath).convert("RGBA")
        array = np.array(image)

        alpha = 1 - (array[:, :image.size[1], 0] / 255)
        # alpha = array[:array.shape[1], :]
        fixed = cv2.resize(alpha, [self.fontSize, self.fontSize])

        imageSize = int(self.fontSize * 1.5)
        full = np.zeros([imageSize, imageSize], dtype=np.float32)
        padding = int(self.fontSize * 0.25)
        full[padding:-padding, padding:-padding] = fixed

        name = os.path.basename(imagePath).removesuffix(".png")

        return name, full


def loadRochesterDescription(descriptionPath):
    name = os.path.basename(descriptionPath)
    text = open(descriptionPath, "r").read().split()

    description = Description(name, text)

    return name, description


# Designed for the MyFonts dataset from Rochester. Does not include actual font files :(
class MyFontsData(QueryData):
    def __init__(self, config, training=False, tokenizer="bert-base-uncased", limit=None):
        self.setTokenizer(tokenizer)

        self.config = config
        self.method = self.config.method if "method" in self.config else "upper"

        imageFunc = Loader(config.fontSize).loadRochesterImage

        images = {}

        imagePaths = glob(os.path.join(config.directory, "fontimage", "*.png"))
        if limit is not None:
            imagePaths = imagePaths[:limit]
        with Pool(processes=2) as pool:
            for i, (name, array) in enumerate(pool.imap(imageFunc, imagePaths, chunksize=1000)):
                images[name] = array
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
        for key in images:
            case = "u" if key[-1] == key[-2] else "l"
            if case == "l":
                continue
            other = key[:-2] + key[-1].lower()

            if other not in images:
                continue

            pairs.append((images[other], images[key]))
            letters.append(key[-2].lower())

            names.append(key[:-3])

        self.names = np.array(names)
        self.pairs = np.array(pairs)
        self.letters = np.array(letters)

        viable = np.isin(names, list(self.descriptions.keys()))
        print(f"{np.mean(viable) * 100:.2f}% of fonts have descriptions")
        self.index = np.arange(len(self.pairs))[viable]

        self.fontMap = {key: key for key in self.descriptions}
        self.fontNum = {key: i for i, key in enumerate(self.descriptions)}
        self.fonts = {key: None for key in self.descriptions}

        # self.fonts Name -> Loaded Font 
    

class CLIPEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # Didn't realize this expects PIL
        # self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.numLayers = 1

        self.outputDimension = config.textProjection if config is not None else None
        if self.outputDimension is not None:
            self.head = nn.Linear(768, self.outputDimension)

    def preprocess(self, x):
        x = x * 255.0
        x = x.clamp(0, 255)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x / 255.0 - mean) / std

        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return x

    def forward(self, x):
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


