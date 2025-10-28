# Script for verifying different models transferability scores
# Need to train several different kinds of models, get several different kinds of metrics


from utils import *
import pickle


def compare(testImageActivations, testTextActivations):
    testScores = {"TransRate": [], "LogME": []}

    for l in range(len(testImageActivations)):
        imageActivations2 = []
        textActivations2 = []
        for key in list(testImageActivations[l].keys()):
            if key not in testTextActivations:
                continue
            imageActivations2.append(testImageActivations[l][key])
            textActivations2.append(testTextActivations[key])

        imageActivations2 = torch.stack(imageActivations2, dim=0).cpu()
        textActivations2 = torch.stack(textActivations2, dim=0).cpu()

        if len(imageActivations2.shape) > 2:
            imageActivations2 = imageActivations2.mean(dim=(2, 3))

        testScores["TransRate"].append(transRate(imageActivations2, textActivations2))
        testScores["LogME"].append(logME(imageActivations2, textActivations2))

    return testScores


checkpoints = ["upper", "lower", "masked", "CLIP"]
textModels = [BertModel, CLIPTextModel]
textModelNames = ["bert-base-uncased", "openai/clip-vit-base-patch32"]

testCharacters = [chr(c) for c in latin]

imageModel, imageConfig = UNet.load(os.path.join("checkpoints", "upper"))

imageConfig.dataset.directory = "google"
dataset = QueryData(imageConfig.dataset)

if not os.path.exists(os.path.join("style", "activations")):
    os.makedirs(os.path.join("style", "activations"), exist_ok=True)

# Get and save activations for each of the image models
for checkpoint in checkpoints:
    if checkpoint == "CLIP":
        imageModel = CLIPEmbedder(None)
    else:
        imageModel, imageConfig = UNet.load(os.path.join("checkpoints", checkpoint))

    if "method" in imageConfig.dataset:
        dataset.method = imageConfig.dataset.method

    # Get all images, upper or lower
    if checkpoint == "CLIP":
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

transRateLayerScores = []
logMELayerScores = []

transScoreMatrix = np.zeros([len(imageActivationPaths), len(textActivationPaths)], dtype=list)
logMEScoreMatrix = np.zeros([len(imageActivationPaths), len(textActivationPaths)], dtype=list)

for i in range(len(imageActivationPaths)):
    with open(imageActivationPaths[i], "rb") as file:
        imageActivations = pickle.load(file)
    for t in range(len(textActivationPaths)):
        with open(textActivationPaths[t], "rb") as file:
            textActivations = pickle.load(file)

        scores = compare(imageActivations, textActivations)

        del textActivations

        torch.cuda.empty_cache()

        transRateLayerScores.append(scores["TransRate"])
        logMELayerScores.append(scores["LogME"])

        transScoreMatrix[i, t] = scores["TransRate"]
        logMEScoreMatrix[i, t] = scores["LogME"]

        plt.suptitle(f"{imageModelNames[i]} -> {textModelNames[t]}")

        plt.subplot(1, 2, 1)
        plt.title(f"TransRate Scores")
        plt.bar(np.arange(len(scores["TransRate"])) + 1, scores["TransRate"])
        plt.ylabel("TransRate")
        plt.xlabel("Layer")
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title(f"LogME Scores")
        plt.bar(np.arange(len(scores["TransRate"])) + 1, scores["LogME"])
        plt.ylabel("LogME")
        plt.xlabel("Layer")
        plt.grid()

        plt.show()

    del imageActivations
    torch.cuda.empty_cache()

transMatrix = [[max(transScoreMatrix[i, t])
                for t in range(len(transScoreMatrix[i]))]
               for i in range(len(transScoreMatrix))]
logMEMatrix = [[max(logMEScoreMatrix[i, t])
                for t in range(len(logMEScoreMatrix[i]))]
               for i in range(len(logMEScoreMatrix))]

transMatrix = np.array(transMatrix)
logMEMatrix = np.array(logMEMatrix)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Model Pre-training TransRate Scores")
ax2.set_title("Model Pre-training LogME Scores")
im1 = ax1.imshow(transMatrix, cmap="viridis")
im2 = ax2.imshow(logMEMatrix, cmap="viridis")
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
ax1.set_xticks(range(len(textModelNames)))
ax1.set_xticklabels(textModelNames)
ax1.set_yticks(range(len(imageModelNames)))
ax1.set_yticklabels(imageModelNames)

ax2.set_xticks(range(len(textModelNames)))
ax2.set_xticklabels(textModelNames)
ax2.set_yticks(range(len(imageModelNames)))
ax2.set_yticklabels(imageModelNames)

ax1.set_xlabel("Text Model")
ax2.set_xlabel("Text Model")
ax1.set_ylabel("Image Model")

plt.show()

