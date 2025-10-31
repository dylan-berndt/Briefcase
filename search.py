from utils import *


textModel = None
imageModel = None
dataset = None


def searchQuery(query, fontVectors):
    textData = Description.tokenizer([query], padding=False, return_tensors="pt")
    embeddedText = textModel(**textData).pooler_output

    fontScores = {}
    # TODO: Refactor to be parallel, one big matrix
    for fontName in fontVectors:
        scores = [embeddedText @ vector for vector in fontVectors[fontName]]
        fontScores[fontName] = np.median(scores), np.mean(scores), np.std(scores), np.min(scores), np.max(scores)

    return fontScores


def generateFontVectors(batchSize=128):
    fontVectors = {}

    inputs = []
    names = []
    for i in range(len(dataset)):
        data = dataset[i]
        inputs.append(data["inputs"])
        names.append(data["name"])

    for i in range(0, len(inputs), batchSize):
        with torch.no_grad():
            j = min(len(inputs) - 1, i + batchSize)
            ips = torch.tensor(inputs[i:j], dtype=torch.float32).squeeze().unsqueeze(-1)
            outputs = imageModel(ips)

            batchNames = names[i:j]

            for n, batchName in enumerate(batchNames):
                if batchName not in fontVectors:
                    fontVectors[batchName] = []

                fontVectors[batchName].append(outputs[n])

    return fontVectors


