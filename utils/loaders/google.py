import math
from PIL import Image, ImageFont
import numpy as np
import os
import cv2
from glob import glob
from bs4 import BeautifulSoup
from .description import *
import spacy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def getAdjectives(filePath, nlp):
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


def loadFolderDescription(args):
    nlp, walk = args
    root, dirs, files = walk
    if not any([file.endswith("METADATA.pb") for file in files]):
        return None
    
    fontNames = []
    fontStyles = []
    caption = None
    
    for file in glob(os.path.join(root, "**", "*.html"), recursive=True):
        if file.endswith(".html"):
            caption = getAdjectives(file, nlp)

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


def loadGoogleDescriptions(directory):
    descriptions = {}

    nlp = spacy.load("en_core_web_sm")

    walks = os.walk(os.path.join(directory, "fonts"))
    roots = [root for root, dirs, files in walks]
    walks = os.walk(os.path.join(directory, "fonts"))
    tasks = [(nlp, walk) for walk in walks]
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        for i, result in enumerate(executor.map(loadFolderDescription, tasks)):
            if result is None:
                continue

            fontNames, fontStyles, caption = result

            if caption is None:
                print(roots[i], fontNames)
                continue

            for j in range(len(fontNames)):
                fontName = fontNames[j]
                fontStyle = fontStyles[j]
                descriptions[fontName] = Description(f"{fontName} {fontStyle}", caption, {fontStyle: 0.2})

            if i % 100 == 0:
                print(f"\rPaths checked: {i + 1}/{len(roots)}", end="")

    print()
                
    tagFile = os.path.join(directory, "fonts", "tags", "all", "families.csv")
    tagDF = pd.read_csv(tagFile, names=["family", "na", "tags", "weight"])
    for family in tagDF.family.unique():
        familyDF = tagDF[tagDF.family == family]
        tags = familyDF.tags.tolist()
        weights = familyDF.weight.tolist()

        tagDict = {}
        for i, tagSet in enumerate(tags):
            tag = tagSet.split("/")[-1]
            tagDict[tag] = weights[i] / 100

        if family not in descriptions:
            continue

        descriptions[family].tags = tagDict
        descriptions[family].fixedSample = descriptions[family]._sample()

    return descriptions