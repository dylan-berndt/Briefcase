# Script for verifying different models transferability scores
# Need to train several different kinds of models, get several different kinds of metrics


from utils import *
import pickle


class Random:
    def __init__(self, dimensions):
        self.numLayers = 1
        self.dimensions = dimensions

    def activations(self, x):
        return [torch.randn([x.shape[0], self.dimensions])]


class Zeroes:
    def __init__(self, dimensions):
        self.numLayers = 1
        self.dimensions = dimensions

    def activations(self, x):
        return [torch.zeros([x.shape[0], self.dimensions])]


def compare(testImageActivations, testTextActivations, testFunctions):
    testScores = {name: [] for name in testFunctions.keys()}

    for l in range(len(testImageActivations)):
        imageActivations2 = []
        textActivations2 = []
        for key in list(testImageActivations[l].keys()):
            if key not in testTextActivations:
                continue
            imageActivations2.append(testImageActivations[l][key])
            textActivations2.append(testTextActivations[key])

        imageActivations2 = torch.stack(imageActivations2, dim=0)
        textActivations2 = torch.stack(textActivations2, dim=0)

        if len(imageActivations2.shape) > 2:
            imageActivations2 = imageActivations2.mean(dim=(2, 3))

        for name in testFunctions.keys():
            testScores[name].append(testFunctions[name](imageActivations2, textActivations2))

    return testScores


checkpoints = ["upper", "lower", "masked", "CLIP", "Random", "Zeroes"]
textModels = [BertModel, CLIPTextModel]
textModelNames = ["bert-base-uncased", "openai/clip-vit-base-patch32"]

testCharacters = [chr(c) for c in latin]

imageModel, imageConfig = UNet.load(os.path.join("checkpoints", "pretrain", "upper"))

imageConfig.dataset.directory = "google"
dataset = QueryData(imageConfig.dataset)

if not os.path.exists(os.path.join("style", "activations")):
    os.makedirs(os.path.join("style", "activations"), exist_ok=True)

# Get and save activations for each of the image models
for checkpoint in checkpoints:
    if checkpoint == "CLIP":
        imageModel = CLIPEmbedder(None)
    elif checkpoint == "Random":
        imageModel = Random(768)
    elif checkpoint == "Zeroes":
        imageModel = Zeroes(768)
    else:
        imageModel, imageConfig = UNet.load(os.path.join("checkpoints", "pretrain", checkpoint))

    if "method" in imageConfig.dataset:
        dataset.method = imageConfig.dataset.method

    # Get all images, upper or lower
    if checkpoint == "CLIP" or checkpoint == "Random" or checkpoint == "Zeroes":
        dataset.method = "masked"

    path = os.path.join("style", "activations", f"{checkpoint} image.pkl")

    if not os.path.exists(path):
        print(f"\nGetting image activations for {checkpoint} model")
        print("=" * 28)
        allImageActivations, _ = imageModelActivations(imageModel, dataset, testCharacters)
        with open(path, "wb") as file:
            pickle.dump(allImageActivations, file)

        del allImageActivations

# Get and save activations for each of the text models
for t in range(len(textModels)):
    # Use lower to prevent even more duplicates of tags being run
    dataset.method = "lower"
    dataset.setTokenizer(textModelNames[t])
    textModel = textModels[t].from_pretrained(textModelNames[t])
    textModel.eval()

    name = textModelNames[t]
    name = name.replace("/", "-")

    path = os.path.join("style", "activations", f"{name} text.pkl")

    if not os.path.exists(path):
        print(f"\nGetting text activations for {name} model")
        print("=" * 28)
        allTextActivations = textModelActivations(textModel, dataset, testCharacters)
        with open(path, "wb") as file:
            pickle.dump(allTextActivations, file)

        del allTextActivations

# The big final matrix
imageActivationPaths = glob(os.path.join("style", "activations", "* image.pkl"))
textActivationPaths = glob(os.path.join("style", "activations", "* text.pkl"))

imageModelNames = [os.path.basename(name).removesuffix(" image.pkl") for name in imageActivationPaths]
textModelNames = [os.path.basename(name).removesuffix(" text.pkl") for name in textActivationPaths]

scoreFunctions = {"TransRate": transRate, "LogME": logME, "LinMSE": linearMSE, "H-Alpha": hAlphaScore}

layerScores = {name: [] for name in scoreFunctions.keys()}
scoreMatrices = {name: np.zeros([len(imageActivationPaths), len(textActivationPaths)], dtype=list)
                 for name in scoreFunctions.keys()}

for i in range(len(imageActivationPaths)):
    with open(imageActivationPaths[i], "rb") as file:
        imageActivations = pickle.load(file)
    for t in range(len(textActivationPaths)):
        with open(textActivationPaths[t], "rb") as file:
            textActivations = pickle.load(file)

        scores = compare(imageActivations, textActivations, scoreFunctions)

        del textActivations

        torch.cuda.empty_cache()

        plt.suptitle(f"{imageModelNames[i]} -> {textModelNames[t]}")

        for n, name in enumerate(scoreFunctions.keys()):
            layerScores[name].append(scores[name])
            scoreMatrices[name][i, t] = scores[name]

            plt.subplot(1, len(scoreFunctions), n + 1)
            plt.title(f"{name} Scores")
            plt.bar(np.arange(len(scores[name])) + 1, scores[name])
            plt.ylabel(name)
            plt.xlabel("Layer")
            plt.grid()

        plt.show()

    del imageActivations
    torch.cuda.empty_cache()


fig, axes = plt.subplots(1, len(scoreFunctions))

for n, name in enumerate(scoreFunctions.keys()):
    matrix = scoreMatrices[name]
    bestMatrix = [[max(matrix[i, t])
                    for t in range(len(matrix[i]))]
                   for i in range(len(matrix))]
    bestMatrix = np.array(bestMatrix)
    im1 = axes[n].imshow(bestMatrix, cmap="viridis")
    fig.colorbar(im1, ax=axes[n])
    axes[n].set_title(f"Pre-training {name} Scores")

    axes[n].set_xlabel("Layer")
    axes[n].set_ylabel(f"{name} Scores")

    axes[n].set_xticks(range(len(textModelNames)))
    axes[n].set_xticklabels(textModelNames)
    axes[n].set_yticks(range(len(imageModelNames)))
    axes[n].set_yticklabels(imageModelNames)

    table = {"name": imageModelNames}
    for t, textName in enumerate(textModelNames):
        table[textName] = bestMatrix[:, t]

    df = pd.DataFrame(table)
    df.to_csv(os.path.join("style", "activations", f"{name}.csv"))

plt.show()

