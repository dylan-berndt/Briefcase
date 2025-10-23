# Utility script to preprocess fonts and descriptions from the Google Fonts repository
# Needs to include name, tags, description
# Then filter description for adjectives and adjective-noun pairs with spacy
# When loading later, can shuffle name, tags, description and randomly include / exclude for robustness

import os

from .model import *
from .data import *
import random
import pandas as pd
from transformers import AutoTokenizer

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

    def __init__(self, name, adjectives, tags=None):
        tags = tags if tags is not None else {}
        
        self.name = name
        self.adjectives = adjectives
        self.tags = tags

    def sample(self):
        sampledAdjectives = random.sample(self.adjectives, int(len(self.adjectives) * random.uniform(0.4, 0.8)))
        sampledTags = [tag for tag, value in self.tags.items() if random.uniform(0, 1) < value]
        joined = ", ".join(sampledAdjectives + sampledTags) + " font"
        if random.uniform(0, 1) > 0.2:
            joined = joined + " named " + self.name

        return "a " + joined

    def __len__(self):
        descriptors = self.adjectives + list(self.tags.keys())
        description = ", ".join(descriptors) + " font named " + self.name
        tokens = Description.tokenizer([description], padding=False, return_tensors="pt")
        return len(tokens["input_ids"][0])


# This is very specifically tailored to the Google Fonts repository
class QueryData(FontData):
    def __init__(self, config, training=False):
        super().__init__(config, training)

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        Description.tokenizer = self.tokenizer

        self.descriptions = {}
        self.fontMap = {}

        lastCaption = None
        for root, dirs, files in os.walk(os.path.join(config.directory, "fonts"), topdown=False):
            for file in files:
                if file.endswith(".html"):
                    filePath = os.path.join(root, file)
                    lastCaption = getAdjectives(filePath)

            for file in files:
                if file.endswith(".otf") or file.endswith(".ttf"):
                    filePath = os.path.join(root, file)
                    try:
                        font = ImageFont.truetype(filePath, 32)
                        fontName, fontStyle = font.getname()

                    except Exception as e:
                        print(filePath, e)
                        continue

                    print(f"\r{fontName} {fontStyle} {lastCaption}", end="")

                    self.fontMap[f"{fontName} {fontStyle}"] = fontName
                    self.descriptions[fontName] = Description(f"{fontName} {fontStyle}", lastCaption, {fontStyle: 0.2})

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

        self.maxLength = max([len(description) for name, description in self.descriptions.items()])

        names = np.array([self.fontMap[name] for name in self.names])
        viable = np.isin(names, list(self.descriptions.keys()))
        print(f"{np.mean(viable) * 100:.2f}% fonts have descriptions")
        self.index = np.arange(len(self.pairs))[viable]

    def __len__(self):
        return len(self.index)

    # TODO: Rework for upper -> lower and masked autoencoder tasks
    def __getitem__(self, i):
        item = self.index[i]
        lower, upper = self.pairs[item]

        lower = torch.tensor(lower, dtype=torch.float32)
        upper = torch.tensor(upper, dtype=torch.float32)

        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)

        fontName = self.fontMap[self.names[item]]
        description = self.descriptions[fontName].sample()
        tokens = self.tokenizer(description, padding='max_length', max_length=self.maxLength, return_tensors="pt")

        character = torch.tensor(characters.index(self.letters[item]), dtype=torch.long)

        return lower, upper, character, tokens["input_ids"][0]
