# Script for verifying different models transferability scores
# Need to train several different kinds of models, get several different kinds of metrics


from utils import *


checkpoints = ["latest"]


for checkpoint in checkpoints:
    model, config = UNet.load(os.path.join("checkpoints", checkpoint))

    config.dataset.directory = "data"
    dataset = QueryData(config.dataset)

    testCharacters = [chr(c) for c in latin]

    allActivations, allImages = imageModelActivations(model, dataset, testCharacters)
