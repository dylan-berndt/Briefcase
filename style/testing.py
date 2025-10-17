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
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# Method for obtaining the correlation between the style applied to the actual images
# and the model's prediction for what style is applied
def batchCorrelation(yTrueNormal, yTrueStyle, yPredNormal, yPredStyle):
    yTrueResidual = yTrueStyle - yTrueNormal
    yPredResidual = yPredStyle - yPredNormal

    result = pearsonr(yTrueResidual.flatten(), yPredResidual.flatten())

    return result.statistic, result.pvalue


def listify(array):
    return [array[i] for i in range(array.shape[0])]


def diagonalCohensD(matrix):
    mask = np.eye(matrix.shape[0], dtype=bool)
    onDiag = matrix[mask]
    offDiag = matrix[~mask]

    diagStd = np.std(onDiag, ddof=1)
    offStd = np.std(offDiag, ddof=1)
    diagMean = np.mean(onDiag)
    offMean = np.mean(offDiag)

    pooledStd = np.sqrt((diagStd ** 2 + offStd ** 2) / 2.0)
    return (diagMean - offMean) / (pooledStd + 1e-6)


def auc(matrix):
    labels = np.eye(matrix.shape[0], dtype=bool)
    return roc_auc_score(labels.flatten(), matrix.flatten())


def diff(matrix):
    mask = np.eye(matrix.shape[0], dtype=bool)
    onDiag = matrix[mask]
    offDiag = matrix[~mask]

    diagMean = np.mean(onDiag)
    offMean = np.mean(offDiag)
    
    return diagMean - offMean


# TODO: Refactor the mess
if __name__ == "__main__":
    # Inconsistencies between VSCode and Pycharm
    cwd = ".." if os.getcwd().endswith("style") else ""

    model, config = UNet.load(os.path.join(cwd, "checkpoints", "latest"))

    config.dataset.directory = os.path.join(cwd, "data")
    dataset = FontData(config.dataset, training=False)

    layers = config.model.layers * 2

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
        references[font] = output.cpu(), targets

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
                                    "styleTarget": listify(targets), "stylePrediction": listify(output.cpu())})

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

    # # Getting all the activations for every font with every character in the latin alphabet
    comparativeFonts = ["Edwardian Script ITC", "Pristina", "Calibri", "Comic Sans MS", "Impact", "Broadway", "Jokerman"]
    testCharacters = [chr(c) for c in latin]
    ablationActivations = []
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
            fontActivations.append(activations)

        ablationActivations.append(fontActivations)
        #TODO: Progress update messages

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

        ax = plt.subplot(height, width, layer + 1)
        ax.set_title(f"Layer {layer + 1} (Mean Diff: {diff(correlation):.2f})")

        correlation = np.flip(correlation, axis=1)
        ax.imshow(correlation, cmap="binary_r", vmin=0, vmax=1)

        ax.set_xticks(range(len(comparativeFonts)), comparativeFonts, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(comparativeFonts)), comparativeFonts[::-1])

        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[0]):
                value = correlation[i, j]
                color = "k" if (value > 0.4) else "w"
                text = ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    plt.suptitle("Layer Activation Cosine Similarity (c1 != c2)")
    plt.show()

    for layer in range(layers):
        mask = np.eye(len(testCharacters), dtype=bool)
        mask = np.expand_dims(mask, (0, 1))

        observe = np.where(mask, ablationMatrix[layer], 0)
        
        correlation = np.sum(observe, axis=(2, 3))
        correlation = correlation / np.sum(mask, axis=(2, 3))

        ax = plt.subplot(height, width, layer + 1)
        ax.set_title(f"Layer {layer + 1} (Mean Diff: {diff(correlation):.2f})")

        correlation = np.flip(correlation, axis=1)
        ax.imshow(correlation, cmap="binary_r", vmin=0, vmax=1)

        ax.set_xticks(range(len(comparativeFonts)), comparativeFonts, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(comparativeFonts)), comparativeFonts[::-1])

        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[0]):
                value = correlation[i, j]
                color = "k" if (value > 0.4) else "w"
                text = ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    plt.suptitle("Layer Activation Cosine Similarity (c1 == c2)")
    plt.show()

    # Only latin characters, prevents character set skew
    mask = np.isin(dataset.letters, np.array(testCharacters))
    pairs = dataset.pairs[mask]
    names = dataset.names[mask]
    letters = dataset.letters[mask]

    # TODO: Refactor for single character
    allActivations = [{} for _ in range(layers)]
    activationCounts = [{} for _ in range(layers)]
    allImages = {}
    batchSize = 128
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

        print(f"\r{j}/{len(pairs)} samples for PCA compiled", end="")

    print()

    # Excluding examples that do not have all the test characters
    # I don't know why some sets have only a subset of the latin characters.
    for name in list(allActivations[0].keys()):
        if activationCounts[0][name] != len(testCharacters):
            print(name, activationCounts[0][name])
            for layer in range(layers):
                del allActivations[layer][name]

    percent = 1
    plt.figure(figsize=(20, 10))
    for layer in range(layers):
        components = PCA(n_components=2)
        data = torch.stack([value for key, value in allActivations[layer].items()], dim=0).cpu().numpy()
        shape = data.shape
        data = np.mean(data, axis=(2, 3))
        # data = data.reshape(data.shape[0], -1)
        print(f"Training PCA from {data.shape[1]} -> 2 dimensions for layer {layer + 1} (Originally {shape})")

        transformed = components.fit_transform(data)
        x, y = transformed[:, 0], transformed[:, 1]

        ax = plt.subplot(height, width, layer + 1)
        ax.scatter(x, y, s=0)

        for i in range(len(x)):
            # Display less of the damned things
            if np.random.rand() > percent:
                continue
            name = list(allActivations[layer].keys())[i]
            image = allImages[name]
            image[image <= 1e-6] = np.nan
            cmap = "binary"
            if "Italic" in name:
                cmap = "viridis"
            if "Bold" in name:
                cmap = "YlGn"
            box = OffsetImage(image, zoom=1, cmap=cmap, interpolation="bilinear", resample=True)
            annotation = AnnotationBbox(box, (x[i], y[i]), xybox=(0, 0), xycoords="data",
                                        boxcoords="offset points", frameon=False)
            
            ax.add_artist(annotation)

        ax.set_title(f"Layer {layer + 1}")

    plt.suptitle("Principal Components per Layer")
    plt.show()

    fontDisplayImages = {name: Image.Image()._new(font.getmask("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for name, font in dataset.fonts.items()}

    for layer in range(layers):
        # components = PCA(n_components=16)
        data = torch.stack([value for key, value in allActivations[layer].items()], dim=0).cpu().numpy()
        shape = data.shape
        data = np.mean(data, axis=(2, 3))
        # data = data.reshape(data.shape[0], -1)
        print(f"Running clustering on {data.shape[0]} samples, {data.shape[-1]} dimensions for layer {layer + 1} (Originally {shape})")

        # transformed = components.fit_transform(data)
        # clustering = OPTICS(min_samples=2, cluster_method='xi', xi=0.01, min_cluster_size=2, metric="cosine")
        # clustering = OPTICS(min_samples=2, cluster_method='dbscan')
        clustering = AgglomerativeClustering(n_clusters=48, metric="cosine", linkage="average")

        clusters = clustering.fit_predict(normalize(data))
        names = np.array(list(allActivations[layer].keys()))

        for clusterID in np.unique(clusters):
            clusterNames = names[clusters == clusterID]
            samples = [fontDisplayImages[name] for name in clusterNames]

            displayHeight = sum([image.size[1] for image in samples]) + (16 * (len(samples) + 1))
            displayWidth = max([image.size[0] for image in samples]) + (32 * 2)

            canvas = Image.new("L", (displayWidth, displayHeight), 0)

            total = 16
            for i in range(len(samples)):
                canvas.paste(samples[i], (32, total))
                total += samples[i].size[1] + 16

            folderPath = os.path.join(cwd, "style", "results", "clustering", f"Layer {layer + 1}")
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)

            imagePath = os.path.join(folderPath, f"Cluster {clusterID}.png")
            canvas.save(imagePath)


    conglomerated = {}
    for layer in range(layers):
        for key in allActivations[layer].keys():
            activation = torch.mean(allActivations[layer][key], dim=(1, 2)).cpu()
            if key not in conglomerated:
                conglomerated[key] = activation
                continue

            conglomerated[key] = torch.cat([conglomerated[key], activation], dim=-1)

    
    data = torch.stack([value for key, value in conglomerated.items()], dim=0).cpu().numpy()
    shape = data.shape
    # data = data.reshape(data.shape[0], -1)
    print(f"Running clustering on {data.shape[0]} samples, {data.shape[-1]} dimensions for entire model")

    # transformed = components.fit_transform(data)
    # clustering = OPTICS(min_samples=2, cluster_method='xi', xi=0.01, min_cluster_size=2, metric="cosine")
    # clustering = OPTICS(min_samples=2, cluster_method='dbscan')
    clustering = AgglomerativeClustering(n_clusters=48, metric="cosine", linkage="average")

    clusters = clustering.fit_predict(normalize(data))
    names = np.array(list(conglomerated.keys()))

    for clusterID in np.unique(clusters):
        clusterNames = names[clusters == clusterID]
        samples = [fontDisplayImages[name] for name in clusterNames]

        displayHeight = sum([image.size[1] for image in samples]) + (16 * (len(samples) + 1))
        displayWidth = max([image.size[0] for image in samples]) + (32 * 2)

        canvas = Image.new("L", (displayWidth, displayHeight), 0)

        total = 16
        for i in range(len(samples)):
            canvas.paste(samples[i], (32, total))
            total += samples[i].size[1] + 16

        folderPath = os.path.join(cwd, "style", "results", "clustering", f"Model")
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        imagePath = os.path.join(folderPath, f"Cluster {clusterID}.png")
        canvas.save(imagePath)

