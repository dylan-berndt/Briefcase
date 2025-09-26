# Script to prove viability of model's analysis of style
# Need to train model with several standardized fonts excluded
# Standardized meaning there exists bold, italicized, serifed, and other variants of the font
# Could also test standard fonts against heavily styled fonts
# Find correlation between the stylistic differences in the ground truth and the predicted images, prove statistical significance

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import *
from scipy.stats import pearsonr


# Method for obtaining the correlation between the style applied to the actual images
# and the model's prediction for what style is applied
def batchCorrelation(yTrueNormal, yTrueStyle, yPredNormal, yPredStyle):
    yTrueResidual = yTrueStyle - yTrueNormal
    yPredResidual = yPredStyle - yPredNormal

    result = pearsonr(yTrueResidual.flatten().numpy(), yPredResidual.flatten().numpy())

    return result.statistic, result.pvalue


if __name__ == "__main__":
    model, config = UNet.load(os.path.join("checkpoints", "latest"))
    dataset = FontData(config)