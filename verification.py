# Script for verifying different models transferability scores
# Need to train several different kinds of models, get several different kinds of metrics


from utils import *


if __name__ == "__main__":
    model, config = UNet.load(os.path.join("checkpoints", "latest"))

    config.dataset.directory = "data"
    dataset = QueryData(config.dataset)

    loader = DataLoader(dataset, batch_size=32)
    inputs = next(iter(loader))
    shape = [i.shape for i in inputs]
    print(shape)
