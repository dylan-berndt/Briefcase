# Script for verifying different models transferability scores
# Need to train several different kinds of models, get several different kinds of metrics


from utils import *
import pickle


def compare(allImageActivations, allTextActivations):
    scores = {"TransRate": [], "LogME": []}

    imageActivations = []
    textActivations = []
    for l in range(len(allImageActivations)):
        for key in list(allImageActivations[l].keys()):
            imageActivations.append(allImageActivations[l][key])
            textActivations.append(allTextActivations[key])

        imageActivations = torch.stack(imageActivations, dim=0).mean(dim=(2, 3)).cpu()
        textActivations = torch.stack(textActivations, dim=0).cpu()

        print(imageActivations.shape)
        print(textActivations.shape)

        scores["TransRate"].append(transRate(imageActivations, textActivations))
        scores["LogME"].append(logME(imageActivations, textActivations))

    return scores


checkpoints = ["upper"]
textModels = [BertModel, CLIPTextModel]
textModelNames = ["bert-base-uncased", "openai/clip-vit-base-patch32"]

testCharacters = [chr(c) for c in latin]

imageModel, imageConfig = UNet.load(os.path.join("checkpoints", "upper"))

imageConfig.dataset.directory = "google"
dataset = QueryData(imageConfig.dataset)

if not os.path.exists("style", "activations"):
    os.makedirs("style", "activations", exist_ok=True)

# Get and save activations for each of the image models
for checkpoint in checkpoints:
    imageModel, imageConfig = UNet.load(os.path.join("checkpoints", checkpoint))
    dataset.method = imageConfig.dataset.method

    path = os.path.join("style", "activations", f"{checkpoint} image.pkl")

    if not os.path.exists(path):
        allImageActivations, _ = imageModelActivations(imageModel, dataset, testCharacters)
        with open(path, "wb") as file:
            pickle.dump(allImageActivations, file)

# Get and save activations for each of the text models
for t in range(len(textModels)):
    # Use lower to prevent even more duplicates of tags being run
    dataset.method = "lower"
    dataset.setTokenizer(textModelNames[t])
    textModel = textModels[t].from_pretrained(textModelNames[t])

    path = os.path.join("style", "activations", f"{textModelNames[t]} text.pkl")

    if not os.path.exists(path):
        allTextActivations = textModelActivations(textModel, dataset, testCharacters)
        with open(path, "wb") as file:
            pickle.dump(allImageActivations, file)

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

plt.title("Model Pre-training TransRate Scores")
bestLayer = np.argmax(averageTransRateScore)
matrix = scoreMatrix[:, :, bestLayer]
ax = plt.imshow(matrix, cmap="viridis")
plt.colorbar()
ax.set_xticks(range(len(textModelNames)))
ax.set_xticklabels(textModelNames)
ax.set_yticks(range(len(imageModelNames)))
ax.set_yticklabels(imageModelNames)

plt.show()
