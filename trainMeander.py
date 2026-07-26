from utils import *

topK = 10
maxSearches = 15

searchHelper = WalkSearch(backbone=os.path.join("checkpoints", "pretrain", "best"), obsVar=1e-3, priorVar=1, obsMargin=0.4, numOptions=8, embeddingName="all")
keys = list(searchHelper.embeddings.keys())
matrix = torch.tensor(np.stack([searchHelper.embeddings[k] for k in keys]), dtype=torch.float32)

norms = matrix.norm(dim=1)
print(norms.mean().item(), norms.std().item(), norms.min().item(), norms.max().item())


class PreferenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.optionEncoder = nn.Sequential(
            nn.Linear(matrix.shape[-1], config.hidden),
            nn.LayerNorm(config.hidden),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden)
        )

        self.lstm = nn.LSTM(config.hidden * 3, config.hidden, num_layers=2, batch_first=True)
    
    def forward(self, o, p, n):
        o = torch.tensor(o, dtype=torch.float32)
        p = torch.tensor(p, dtype=torch.float32)
        n = torch.tensor(n, dtype=torch.float32)

        o = self.optionEncoder(o).mean(dim=-2)
        p = self.optionEncoder(p)
        n = self.optionEncoder(n)

        x = torch.stack([o, p, n], dim=-1)
        x, _ = self.lstm(x)

        return x[:, -1]
    

class TrialDataset(Dataset):
    def __init__(self):
        self.options = []
        self.positives = []
        self.negatives = []
        self.trialNum = []

    def append(self, o, p, n, t):
        self.options.append(o)
        self.positives.append(p)
        self.negatives.append(n)
        self.trialNum.append(t)

    def __len__(self):
        return len(self.options)
    
    def __getitem__(self, i):
        return self.options[i], self.positives[i], self.negatives[i], self.trialNum[i]



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


margin = estimateObsMargin(searchHelper)
searchHelper = WalkSearch(backbone=os.path.join("checkpoints", "pretrain", "best"), obsVar=1e-3, priorVar=1, obsMargin=margin, numOptions=8, embeddingName="all")


def obtainTrials():
    dataset = TrialDataset()

    for i in range(10000):
        optionHistory = []
        positiveHistory = []
        negativeHistory = []

        searchHelper.initializeLearner(obsVar=1e-3)
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

            if k <= topK:
                found = True
                break

            scores = [(target @ option[1]).item() for option in searchHelper.options]
            positive = searchHelper.options[scores.index(max(scores))]
            negative = searchHelper.options[scores.index(min(scores))]

            optionHistory.append(searchHelper.options)
            positiveHistory.append(positive)
            negativeHistory.append(negative)

            dataset.append(optionHistory, positiveHistory, negativeHistory, i)

            searchHelper.updateLocation(positive, negative)

            searches += 1

            if searches >= maxSearches:
                break

        print(f"\r{i}/10000", end="")

    print()

    return dataset


data = obtainTrials()
model = PreferenceModel(Config(hidden=32))

trajectories = []