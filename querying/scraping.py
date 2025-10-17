# Utility script to preprocess fonts and descriptions from the Google Fonts repository
# Needs to include name, tags, description
# Then filter description for adjectives and adjective-noun pairs with spacy
# When loading later, can shuffle name, tags, description and randomly include / exclude for robustness

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import *
import random
import pandas as pd

from bs4 import BeautifulSoup
import spacy

nlp = spacy.load("en_core_web_sm")


def getAdjectives(filePath):
    with open(filePath, "r") as file:
        data = file.read()
        soup = BeautifulSoup(data, 'html.parser')
        text = soup.get_text()

        # TODO: Include adjective-noun phrases?
        adjectives = [token.text for token in nlp(text) if token.pos_ == "ADJ"]
        return adjectives
    

class Description:
    def __init__(self, name, adjectives, tags=None):
        tags = tags if tags is not None else []
        
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


# This is very specifically tailored to the Google Fonts repository
class QueryData(FontData):
    def __init__(self, config, training=False):
        super().__init__(config, training)

        self.descriptions = {}

        lastCaption = None
        for root, dirs, files in os.walk(os.path.join(config.directory, "fonts")):
            for file in files:
                if file.endswith(".html"):
                    filePath = os.path.join(root, file)
                    lastCaption = getAdjectives(filePath)

            for file in files:
                if file.endswith(".otf") or file.endswith(".ttf"):
                    filePath = os.path.join(root, file)
                    font = ImageFont.truetype(filePath, 32)
                    fontName, fontStyle = font.getname()

                    self.descriptions[f"{fontName}"] = Description(f"{fontName} {fontStyle}", lastCaption, [fontStyle])

        tagFile = os.path.join(config.directory, "fonts", "tags", "families.csv")
        tagDF = pd.read_csv(tagFile, names=["family", "na", "tags", "weight"])
        for family in tagDF.family.unique():
            familyDF = tagDF[tagDF.family == family]
            tags = familyDF.tags.tolist()
            weights = familyDF.weight.tolist()

            tagDict = {}
            for i, tagSet in enumerate(tags):
                tag = tagSet.split("/")[-1]
                tagDict[tag] = weights[i] / 100

            self.descriptions[family].tags = tagDict

    def __getitem__(self, i):
        pass


if __name__ == "__main__":
    cwd = ".." if os.getcwd().endswith("style") else ""
    model, config = UNet.load(os.path.join(cwd, "checkpoints", "latest"))

    config.dataset.directory = os.path.join(cwd, "data")
    dataset = QueryData(config.dataset)
