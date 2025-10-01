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

    result = pearsonr(yTrueResidual.flatten().numpy(), yPredResidual.flatten().numpy())

    return result.statistic, result.pvalue


def listify(array):
    return [array[i] for i in range(array.shape[0])]


if __name__ == "__main__":
    model, config = UNet.load(os.path.join("..", "checkpoints", "latest"))

    config.dataset.directory = os.path.join("..", "data")
    dataset = FontData(config.dataset)

    standardFonts = {
        "Calibri": ["Regular", "Bold Italic", "Light Italic", "Light", "Italic", "Bold"],
        "Arial": ["Regular", "Rounded", "Narrow", "Narrow Italic", "Narrow Bold Italic", "Narrow Bold",
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
        inputs = dataset.pairs[namespace][: 0]
        targets = dataset.pairs[namespace][: 1]
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)
        output, classification = model(inputs)
        references[font] = output, targets

    for font in namespaces:
        series = {}
        for style in namespaces[font]:
            if style == "reference":
                continue

            namespace = namespaces[font][style]
            inputs = dataset.pairs[namespace][: 0]
            targets = dataset.pairs[namespace][: 1]
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)
            output, classification = model(inputs)

            referenceOrder = dataset.letters[namespaces[font]["reference"]]
            styleOrder = dataset.letters[namespace]

            refTarget, refPred = references[font]

            referenceDF = pd.DataFrame({"letter": referenceOrder,
                                        "refTarget": listify(refTarget), "refPrediction": listify(refPred)})
            styleDF = pd.DataFrame({"letter": styleOrder,
                                    "styleTarget": listify(targets), "stylePrediction": listify(output)})

            joined = pd.merge(styleDF, referenceDF, how="inner", on="letter")
            print(font, style, len(joined))

            series[f"{style} ({len(joined)})"] = [
                batchCorrelation(joined.refTarget.to_numpy()[i], joined.styleTarget.to_numpy()[i],
                                 joined.refPrediction.to_numpy()[i], joined.stylePrediction.to_numpy()[i])
                for i in range(len(joined))
            ]

        x = series.keys()
        y = [series[key] for key in x]

        ax = plt.subplot(1, 1, 1)
        ax.boxplot(y)
        ax.set_xticklabels(x)
        ax.set_xlabel("Style")
        ax.set_ylabel("Correlation")
        ax.grid()

        plt.title(f"{font} Style Correlations")
        plt.show()

