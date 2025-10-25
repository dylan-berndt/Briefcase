# Script for verifying different models transferability scores
# Need to train several different kinds of models, get several different kinds of metrics


from utils import *
import pickle


def compare(testImageActivations, testTextActivations):
    testScores = {"TransRate": [], "LogME": []}

    for l in range(len(testImageActivations)):
        imageActivations = []
        textActivations = []
        for key in list(testImageActivations[l].keys()):
            if key not in testTextActivations:
                continue
            imageActivations.append(testImageActivations[l][key])
            textActivations.append(testTextActivations[key])

        imageActivations = torch.stack(imageActivations, dim=0).mean(dim=(2, 3)).cpu()
        textActivations = torch.stack(textActivations, dim=0).cpu()

        testScores["TransRate"].append(transRate(imageActivations, textActivations))
        testScores["LogME"].append(logME(imageActivations, textActivations))

    return testScores


checkpoints = ["upper"]
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
    imageModel, imageConfig = UNet.load(os.path.join("checkpoints", checkpoint))
    if "method" in imageConfig.dataset:
        dataset.method = imageConfig.dataset.method

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

scoreMatrix = np.zeros([len(imageActivationPaths), len(textActivationPaths), imageModel.numLayers])

for i in range(len(imageActivationPaths)):
    with open(imageActivationPaths[i], "rb") as file:
        imageActivations = pickle.load(file)
    for t in range(len(textActivationPaths)):
        with open(textActivationPaths[i], "rb") as file:
            textActivations = pickle.load(file)

        scores = compare(imageActivations, textActivations)

        transRateLayerScores.append(scores["TransRate"])
        logMELayerScores.append(scores["LogME"])

        scoreMatrix[i, t] = scores["TransRate"]

        plt.subplot(1, 2, 1)
        plt.title(f"TransRate Scores {imageModelNames[i]} -> {textModelNames[t]}")
        plt.bar(range(imageModel.numLayers), scores["TransRate"])
        plt.ylabel("TransRate")
        plt.xlabel("Layer")
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title(f"LogME Scores {imageModelNames[i]} -> {textModelNames[t]}")
        plt.bar(range(imageModel.numLayers), scores["LogME"])
        plt.ylabel("LogME")
        plt.xlabel("Layer")
        plt.grid()

        plt.show()

averageTransRateScore = np.array(transRateLayerScores).mean(axis=0)
averageLogMEScore = np.array(logMELayerScores).mean(axis=0)

plt.subplot(1, 2, 1)
plt.title(f"Average Layer TransRate Scores")
plt.bar(range(imageModel.numLayers), averageTransRateScore)
plt.ylabel("TransRate")
plt.xlabel("Layer")
plt.grid()

plt.subplot(1, 2, 2)
plt.title(f"Average Layer LogME Scores")
plt.bar(range(imageModel.numLayers), averageLogMEScore)
plt.ylabel("LogME")
plt.xlabel("Layer")
plt.grid()

plt.show()

bestLayer = np.argmax(averageTransRateScore)
matrix = scoreMatrix[:, :, bestLayer]
fig, ax = plt.subplots()
plt.title("Model Pre-training TransRate Scores")
plt.imshow(matrix, cmap="viridis")
plt.colorbar()
ax.set_xticks(range(len(textModelNames)))
ax.set_xticklabels(textModelNames)
ax.set_yticks(range(len(imageModelNames)))
ax.set_yticklabels(imageModelNames)

ax.set_xlabel("Text Model")
ax.set_ylabel("Image Model")

plt.show()
