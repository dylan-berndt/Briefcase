# Utility script to preprocess fonts and descriptions from the Google Fonts repository
# Needs to include name, tags, description
# Then filter description for adjectives and adjective-noun pairs with spacy
# When loading later, can shuffle name, tags, description and randomly include / exclude for robustness

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





