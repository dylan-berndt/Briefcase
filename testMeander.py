from utils import *

epsilon = 0.01

rates = [0.3, 0.1, 0.01, 0.0001]
results = [[] for _ in range(len(rates))]

searchHelper = MeanderSearch(backbone=os.path.join("checkpoints", "pretrain", "best"), learningRate=0.3)

for r, rate in enumerate(rates):
    status = []

    for i in range(1000):
        searchHelper.initializeLearner(rate)
        names = list(searchHelper.embeddings.keys())
        fontName = random.choice(names)
        target = searchHelper.embeddings[fontName]
        targetT = torch.tensor(target, dtype=torch.float32)

        searches = 1
        found = False
        while not found:
            score = (target @ nn.functional.normalize(searchHelper.location, dim=-1)).item()
            if 1 - score < epsilon:
                found = True
                break

            scores = [(target @ option[1]).item() for option in searchHelper.options]
            positive = searchHelper.options[scores.index(max(scores))]
            negative = searchHelper.options[scores.index(min(scores))]

            searchHelper.updateLocation(positive, negative)

            searches += 1

            if searches >= 25:
                break

        status.append(found)
        results[r].append(searches)

        print(f"\r{i}/1000 | Learning Rate: {rate:.3f} | Success Rate: {(sum(status) / len(status)) * 100:.2f}%", end="")

    print()

for r, rate in enumerate(rates):
    plt.title(f"Learning Rate {rate}")
    plt.hist(results[r])
    plt.show()