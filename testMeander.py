from utils import *

topK = 10
maxSearches = 15

rates = [0.3, 0.1, 0.01, 0.0001]
results = [[] for _ in range(len(rates))]
trajectories = [[] for _ in range(len(rates))]

searchHelper = MeanderSearch(backbone=os.path.join("checkpoints", "pretrain", "best"), learningRate=0.3)
keys = list(searchHelper.embeddings.keys())
matrix = torch.tensor(np.stack([searchHelper.embeddings[k] for k in keys]), dtype=torch.float32)

for r, rate in enumerate(rates):
    status = []
    trajectory = []

    for i in range(1000):
        searchHelper.initializeLearner(rate)
        names = list(searchHelper.embeddings.keys())
        fontName = random.choice(names)
        target = searchHelper.embeddings[fontName]
        targetT = torch.tensor(target, dtype=torch.float32)

        searches = 1
        found = False
        while not found:
            location = nn.functional.normalize(searchHelper.location, dim=-1)
            scores = (matrix @ location).numpy()
            targetScore = (target @ location).item()
            k = int((scores > targetScore).sum())

            trajectory.append(k)

            if k <= topK:
                found = True
                break

            scores = [(target @ option[1]).item() for option in searchHelper.options]
            positive = searchHelper.options[scores.index(max(scores))]
            negative = searchHelper.options[scores.index(min(scores))]

            searchHelper.updateLocation(positive, negative)

            searches += 1

            if searches >= maxSearches:
                break

        for i in range(maxSearches - len(trajectory)):
            trajectory.append(1)

        status.append(found)
        results[r].append(searches)
        trajectories[r].append(trajectory)

        print(f"\r{i}/1000 | Learning Rate: {rate:.3f} | Success Rate: {(sum(status) / len(status)) * 100:.2f}%", end="")

    print()


trajectories = np.array(trajectories)

for r, rate in enumerate(rates):
    plt.title(f"Learning Rate {rate}")
    plt.hist(results[r])
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(f"Median Trajectory {rate}")
    ax.boxplot(np.transpose(trajectories[r]))
    ax.set_yscale("log")

    plt.show()
    