from utils import *

topK = 10
maxSearches = 15

rates = [1e-1, 1e-2, 1e-3, 1e-4]
results = [[] for _ in range(len(rates))]
trajectories = [[] for _ in range(len(rates))]

searchHelper = WalkSearch(backbone=os.path.join("checkpoints", "pretrain", "best"), obsVar=1e-3, priorVar=1, obsMargin=0.4, numOptions=8, embeddingName="all")
keys = list(searchHelper.embeddings.keys())
matrix = torch.tensor(np.stack([searchHelper.embeddings[k] for k in keys]), dtype=torch.float32)

norms = matrix.norm(dim=1)
print(norms.mean().item(), norms.std().item(), norms.min().item(), norms.max().item())


def estimateObsMargin(searchHelper, trials=300, roundsPerTrial=6):
    alignments = []

    for _ in range(trials):
        searchHelper.initializeLearner()
        fontName = random.choice(keys)
        target = torch.tensor(searchHelper.embeddings[fontName], dtype=torch.float32)
        targetUnit = nn.functional.normalize(target, dim=-1)

        for _ in range(roundsPerTrial):
            location = nn.functional.normalize(searchHelper.location, dim=-1)
            # residual direction u_k: component of target orthogonal to location
            proj = targetUnit - (targetUnit @ location) * location
            u_k = nn.functional.normalize(proj, dim=-1)

            scores = [(target @ torch.tensor(o[1], dtype=torch.float32)).item()
                      for o in searchHelper.options]
            positive = searchHelper.options[scores.index(max(scores))]
            negative = searchHelper.options[scores.index(min(scores))]

            optionsMatrix = nn.functional.normalize(
                torch.tensor(np.stack([o[1] for o in searchHelper.options]), dtype=torch.float32), dim=-1)
            p = nn.functional.normalize(torch.tensor(positive[1], dtype=torch.float32), dim=-1)
            n = nn.functional.normalize(torch.tensor(negative[1], dtype=torch.float32), dim=-1)
            axis = p - n
            weights = optionsMatrix @ axis
            a = weights @ optionsMatrix          # same centroid-weighted vector updateLocation uses

            alignments.append((a @ u_k).item())

            searchHelper.updateLocation(positive, negative)

    alignments = np.array(alignments)
    print(f"mean(a·u_k) = {alignments.mean():.4f}  (this is your obsMargin)")
    print(f"std(a·u_k)  = {alignments.std():.4f}  (this is v*, useful for the s*^2 formulas)")
    return alignments.mean()


def calibrationCheck(searchHelper, obsVar, trials=200, roundsPerTrial=8):
    keys = searchHelper.matrixKeys
    matrix = searchHelper.matrix
    nis = []

    for _ in range(trials):
        searchHelper.initializeLearner(obsVar=obsVar)
        fontName = random.choice(keys)
        target = torch.tensor(searchHelper.embeddings[fontName], dtype=torch.float32)

        for _ in range(roundsPerTrial):
            scores = [(target @ torch.tensor(o[1], dtype=torch.float32)).item()
                      for o in searchHelper.options]
            positive = searchHelper.options[scores.index(max(scores))]
            negative = searchHelper.options[scores.index(min(scores))]

            p = nn.functional.normalize(torch.tensor(positive[1], dtype=torch.float32), dim=-1)
            n = nn.functional.normalize(torch.tensor(negative[1], dtype=torch.float32), dim=-1)
            a = p - n

            yPred = searchHelper.obsMargin - a @ searchHelper.location
            sPred = a @ searchHelper.covariance @ a + searchHelper.obsVar

            nis.append((yPred.item() ** 2) / sPred.item())

            searchHelper.updateLocation(positive, negative)

    nis = np.array(nis)
    print(f"obsVar={obsVar}: mean NIS={nis.mean():.3f} (well-calibrated ~= 1.0)")
    return nis.mean()


margin = estimateObsMargin(searchHelper)
searchHelper = WalkSearch(backbone=os.path.join("checkpoints", "pretrain", "best"), obsVar=1e-3, priorVar=1, obsMargin=margin, numOptions=8, embeddingName="all")

# for obsVar in rates:
#     calibrationCheck(searchHelper, obsVar)


for r, rate in enumerate(rates):
    status = []

    for i in range(1000):
        trajectory = []
        searchHelper.initializeLearner(obsVar=rate)
        names = list(searchHelper.embeddings.keys())
        fontName = random.choice(names)
        target = searchHelper.embeddings[fontName]
        targetT = torch.tensor(target, dtype=torch.float32)

        searches = 1
        found = False
        while not found:
            location = nn.functional.normalize(searchHelper.location, dim=-1)
            with torch.no_grad():
                scores = (matrix @ location).numpy()
                targetScore = (targetT @ location).item()
            k = int((scores > targetScore).sum())
            # print(searchHelper.location.norm(p=2).item(), targetScore, k)

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

        for _ in range(maxSearches - len(trajectory)):
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
    ax.boxplot(trajectories[r])
    ax.set_yscale("log")

    plt.show()
    