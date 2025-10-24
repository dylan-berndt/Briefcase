# Script for verifying different models transferability scores
# Need to train several different kinds of models, get several different kinds of metrics


from utils import *


checkpoints = ["latest"]

for checkpoint in checkpoints:
    imageModel, imageConfig = UNet.load(os.path.join("checkpoints", checkpoint))

    imageConfig.dataset.directory = "google"
    dataset = QueryData(imageConfig.dataset)

    testCharacters = [chr(c) for c in latin]

    allImageActivations, allImages = imageModelActivations(imageModel, dataset, testCharacters)

    # textModel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    textModel = BertModel.from_pretrained("bert-base-uncased")

    allTextActivations = textModelActivations(textModel, dataset, testCharacters)

    imageActivations = []
    textActivations = []
    for key in list(allImageActivations[0].keys()):
        imageActivations.append(allImageActivations[-1][key].mean(dim=(2, 3)))
        textActivations.append(allTextActivations[key])

    imageActivations = torch.stack(imageActivations, dim=0).cpu()
    textActivations = torch.stack(textActivations, dim=0).cpu()

    print(transRate(imageActivations, textActivations))


