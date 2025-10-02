# Script to prove viability of model's analysis of style
# Need to train model with several standardized fonts excluded
# Standardized meaning there exists bold, italicized, serifed, and other variants of the font
# Could also test standard fonts against heavily styled fonts
# Find correlation between the stylistic differences in the ground truth and the predicted images
# Prove statistical significance

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import *
from scipy.stats import pearsonr

import pandas as pd
import matplotlib.pyplot as plt


# Method for obtaining the correlation between the style applied to the actual images
# and the model's prediction for what style is applied
def batchCorrelation(yTrueNormal, yTrueStyle, yPredNormal, yPredStyle):
    yTrueResidual = yTrueStyle - yTrueNormal
    yPredResidual = yPredStyle - yPredNormal

    result = pearsonr(yTrueResidual.flatten(), yPredResidual.flatten())

    return result.statistic, result.pvalue


def listify(array):
    return [array[i] for i in range(array.shape[0])]


# TODO: Refactor the mess
if __name__ == "__main__":
    model, config = UNet.load(os.path.join("checkpoints", "latest"))

    config.dataset.directory = os.path.join("data")
    dataset = FontData(config.dataset, training=False)

    standardFonts = {
        "Calibri": ["Regular", "Bold Italic", "Light Italic", "Light", "Italic", "Bold"],
        "Arial": ["Regular", "Narrow", "Narrow Italic", "Narrow Bold Italic", "Narrow Bold",
                  "Italic", "Bold Italic", "Bold"],
        "Times New Roman": ["Regular", "Bold", "Bold Italic", "Italic"]
    }

    # Compiling all available samples for the standard fonts
    namespaces = {font: {"reference": None} for font in standardFonts}
    for font in standardFonts:
        for style in standardFonts[font]:
            space = dataset.names == f"{font} {style}"
            if namespaces[font]["reference"] is None:
                namespaces[font]["reference"] = space
                continue
            namespaces[font][style] = space

    # Getting reference "unstyled" glyph predictions and targets
    references = {}
    for font in namespaces:
        namespace = namespaces[font]["reference"]
        inputs = dataset.pairs[namespace][:, 0]
        targets = dataset.pairs[namespace][:, 1]
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32).squeeze().unsqueeze(-1)
            output, classification = model(inputs)
        references[font] = output, targets

    for font in namespaces:
        series = {}
        for style in namespaces[font]:
            if style == "reference":
                continue

            namespace = namespaces[font][style]
            inputs = dataset.pairs[namespace][:, 0]
            targets = dataset.pairs[namespace][:, 1]
            with torch.no_grad():
                inputs = torch.tensor(inputs, dtype=torch.float32).squeeze().unsqueeze(-1)
                output, classification = model(inputs)

            referenceOrder = dataset.letters[namespaces[font]["reference"]]
            styleOrder = dataset.letters[namespace]

            refPred, refTarget = references[font]

            referenceDF = pd.DataFrame({"letter": referenceOrder,
                                        "refTarget": listify(refTarget), "refPrediction": listify(refPred)})
            styleDF = pd.DataFrame({"letter": styleOrder,
                                    "styleTarget": listify(targets), "stylePrediction": listify(output)})

            joined = pd.merge(styleDF, referenceDF, how="inner", on="letter")

            series[f"{style} ({len(joined)})"] = [
                batchCorrelation(joined.refTarget.to_numpy()[i], joined.styleTarget.to_numpy()[i],
                                 joined.refPrediction.to_numpy()[i], joined.stylePrediction.to_numpy()[i])[0]
                for i in range(len(joined))
            ]

        x = series.keys()
        y = [series[key] for key in x]

        ax = plt.subplot(1, 1, 1)
        ax.boxplot(y)
        ax.set_xticklabels(x)
        # ax.set_xticks(range(1, len(x) + 1), x, rotation=45)
        ax.set_xlabel("Style")
        ax.set_ylabel("Correlation")
        ax.grid()

        plt.title(f"{font} Style Correlations")
        plt.show()

    # Getting all the activations for every font with every character in the latin alphabet
    comparativeFonts = ["Edwardian Script ITC", "Calibri", "Comic Sans MS", "Broadway", "Jokerman", "Wide Latin"]
    testCharacters = [chr(c) for c in latin]
    ablationActivations = []
    layers = None
    for font in comparativeFonts:
        fontActivations = []
        for character in testCharacters:
            # Just plainly wasteful, but keeps things in order without sorting so idc
            if np.sum(np.bitwise_and(dataset.names == f"{font} Regular", dataset.letters == character)) == 0:
                print(font, character)
            pair = dataset.pairs[np.bitwise_and(dataset.names == f"{font} Regular", dataset.letters == character)]
            with torch.no_grad():
                inputs = torch.tensor(pair[:, 0], dtype=torch.float32).squeeze().unsqueeze(0).unsqueeze(-1)
                activations = model.activations(inputs)
            layers = len(activations)
            fontActivations.append(activations)

        ablationActivations.append(fontActivations)

    # All possible combinations of layer activation similarities
    # Yes this is perhaps the single most revolting bit of code I have ever written
    ablationMatrix = np.zeros([layers, len(comparativeFonts), len(comparativeFonts), len(testCharacters), len(testCharacters)])
    for f1 in range(len(comparativeFonts)):
        for f2 in range(len(comparativeFonts)):
            for c1 in range(len(testCharacters)):
                for c2 in range(len(testCharacters)):
                    for l in range(layers):
                        x = ablationActivations[f1][c1][l].flatten()
                        y = ablationActivations[f2][c2][l].flatten()
                        ablationMatrix[l, f1, f2, c1, c2] = nn.functional.cosine_similarity(x, y, dim=0).item()

    width = int(math.ceil(math.sqrt(layers * 2)))
    height = int(math.ceil(layers / width))
    for layer in range(layers):
        mask = np.eye(len(testCharacters), dtype=bool)
        mask = np.expand_dims(mask, (0, 1))

        observe = np.where(mask, 0, ablationMatrix[layer])
        
        # Weighted sum (char 1 = char 2 has a weight of 0)
        correlation = np.sum(observe, axis=(2, 3))
        correlation = correlation / np.sum(~mask, axis=(2, 3))
        correlation = np.flip(correlation, axis=0)

        ax = plt.subplot(height, width, layer + 1)
        ax.set_title(f"Layer {layer + 1}")
        ax.imshow(correlation, cmap="binary_r", vmin=0, vmax=1)
        # ax.colorbar()
        ax.set_xticks(range(len(comparativeFonts)), comparativeFonts, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(comparativeFonts)), comparativeFonts[::-1])

        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[0]):
                value = correlation[i, j]
                color = "k" if (value > np.percentile(correlation, 40)) else "w"
                text = ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    plt.suptitle("Layer Activation Correlations (c1 != c2)")
    plt.show()

    for layer in range(layers):
        mask = np.eye(len(testCharacters), dtype=bool)
        mask = np.expand_dims(mask, (0, 1))

        observe = np.where(mask, ablationMatrix[layer], 0)
        
        correlation = np.sum(observe, axis=(2, 3))
        correlation = correlation / np.sum(mask, axis=(2, 3))
        correlation = np.flip(correlation, axis=0)

        ax = plt.subplot(height, width, layer + 1)
        ax.set_title(f"Layer {layer + 1}")
        ax.imshow(correlation, cmap="binary_r", vmin=0, vmax=1)
        # ax.colorbar()
        ax.set_xticks(range(len(comparativeFonts)), comparativeFonts, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(comparativeFonts)), comparativeFonts[::-1])

        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[0]):
                value = correlation[i, j]
                color = "k" if (value > np.percentile(correlation, 40)) else "w"
                text = ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    plt.suptitle("Layer Activation Correlations (c1 == c2)")
    plt.show()

