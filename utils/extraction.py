# This is where that refactoring of style/testing should go
# The script should extract all the activations from the model for each testing scenario
# Can be reused in transferability estimation

from .model import *
from .data import *
from .querying import *


def imageModelActivations(model, dataset, testCharacters, batchSize=128):
    layers = model.numLayers

    inputs = []
    names = []
    letters = []
    for i in range(len(dataset)):
        data = dataset[i]
        inputs.append(data["inputs"].cpu())
        names.append(data["name"])
        letters.append(data["letter"])

    inputs, names, letters = np.array(inputs), np.array(names), np.array(letters)

    mask = np.isin(letters, np.array(testCharacters))
    inputs = inputs[mask]
    names = names[mask]
    letters = letters[mask]

    allActivations = [{} for _ in range(layers)]
    activationCounts = [{} for _ in range(layers)]
    allImages = {}
    for i in range(0, len(inputs), batchSize):
        with torch.no_grad():
            j = min(len(inputs) - 1, i + batchSize)
            ips = torch.tensor(inputs[i:j], dtype=torch.float32).squeeze().unsqueeze(-1)
            activations = model.activations(ips)

            batchNames = names[i:j]
            batchImages = inputs[i:j]
            batchLetters = letters[i:j]

            for n, name in enumerate(batchNames):
                for layer in range(layers):
                    # Running average of activation due to memory usage
                    if name in allActivations[layer]:
                        num = activationCounts[layer][name]
                        allActivations[layer][name] = (allActivations[layer][name] * num + activations[layer][n])\
                                                      / (num + 1)
                        activationCounts[layer][name] += 1
                    else:
                        allActivations[layer][name] = activations[layer][n]
                        activationCounts[layer][name] = 1

                    # Add reference image to dictionary
                    if name not in allImages and batchLetters[n] == "b":
                        allImages[name] = batchImages[n]

        print(f"\r{j}/{len(inputs)} image activation samples compiled", end="")

    print()

    # Excluding examples that do not have all the test characters
    # I don't know why some sets have only a subset of the latin characters.
    for name in list(allActivations[0].keys()):
        if activationCounts[0][name] != len(testCharacters):
            print(name, activationCounts[0][name])
            for layer in range(layers):
                del allActivations[layer][name]

    return allActivations, allImages


def textModelActivations(model, dataset, testCharacters, batchSize=128):
    names = []
    letters = []
    for i in range(len(dataset)):
        data = dataset[i]
        names.append(data["name"])
        letters.append(data["letter"])

    names, letters = np.array(names), np.array(letters)

    loader = DataLoader(dataset, batch_size=batchSize, collate_fn=dataset.collate)

    allActivations = {}
    activationCounts = {}

    # I realize that this runs the same description for each letter in the test set,
    # but also I don't care and maybe it will average the tag sampling in a neat way
    for i, batch in enumerate(loader):
        with torch.no_grad():
            _, _, _, text = batch
            outputs = model(**text)
            pooled = outputs.pooler_output

            j = i * batchSize
            k = min(len(dataset) - 1, (i + 1) * batchSize)

            batchNames = names[j:k]

            for n, name in enumerate(batchNames):
                # Running average of activation due to memory usage
                if name in allActivations:
                    num = activationCounts[name]
                    allActivations[name] = (allActivations[name] * num + pooled[n]) / (num + 1)
                    activationCounts[name] += 1
                else:
                    allActivations[name] = pooled[n]
                    activationCounts[name] = 1

        print(f"\r{j}/{len(loader) * batchSize} text activation samples compiled", end="")

    print()

    # Excluding examples that do not have all the test characters
    # I don't know why some sets have only a subset of the latin characters.
    for name in list(allActivations.keys()):
        if activationCounts[name] != len(testCharacters):
            print(name, activationCounts[name])
            del allActivations[name]

    return allActivations
