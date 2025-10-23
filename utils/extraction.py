# This is where that refactoring of style/testing should go
# The script should extract all the activations from the model for each testing scenario
# Can be reused in transferability estimation

from .model import *
from .data import *


def imageModelActivations(model, dataset, testCharacters, batchSize=128):
    layers = model.numLayers

    mask = np.isin(dataset.letters, np.array(testCharacters))

    # TODO: Rework for upper -> lower and masked autoencoder tasks
    pairs = dataset.pairs[mask]
    names = dataset.names[mask]
    letters = dataset.letters[mask]

    allActivations = [{} for _ in range(layers)]
    activationCounts = [{} for _ in range(layers)]
    allImages = {}
    for i in range(0, len(pairs), batchSize):
        with torch.no_grad():
            j = min(len(pairs) - 1, i + batchSize)
            inputs = torch.tensor(pairs[i:j, 0], dtype=torch.float32).squeeze().unsqueeze(-1)
            activations = model.activations(inputs)

            batchNames = names[i:j]
            batchImages = pairs[i:j, 0]
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

        print(f"\r{j}/{len(pairs)} activation samples compiled", end="")

    print()

    # Excluding examples that do not have all the test characters
    # I don't know why some sets have only a subset of the latin characters.
    for name in list(allActivations[0].keys()):
        if activationCounts[0][name] != len(testCharacters):
            print(name, activationCounts[0][name])
            for layer in range(layers):
                del allActivations[layer][name]

    return allActivations, allImages

