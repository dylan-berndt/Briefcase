import torch
import torch.nn as nn


# https://github.com/kreasof-ai/sigreg
def sigreg_weak_loss(x, sketch_dim=64):
    """
    Forces Covariance(x) ~ Identity.
    Matches the 2nd Moment (Spherical Cloud).
    """
    N, C = x.size()
    # 1. Sketching (Optional for C=512, but good for consistency)
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T  # [N, sketch_dim]
    else:
        sketch_dim = C

    # 2. Centering & Covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 3. Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # 4. Off-diagonal suppression + Diagonal maintenance
    return torch.norm(cov - target, p='fro')


class MoCoQueue:
    def __init__(self, dim: int, size: int = 8192):
        self.queue = nn.functional.normalize(torch.randn(size, dim), dim=-1)
        self.families = torch.full((size,), fill_value=-1, dtype=torch.long)
        self.ptr = 0
        self.size = size

    def get(self):
        return self.queue.clone(), self.families.clone()

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, families: torch.Tensor):
        device = embeddings.device
        self.queue = self.queue.to(device)
        n = embeddings.shape[0]
        # Wrap around if needed
        slots = torch.arange(self.ptr, self.ptr + n) % self.size
        self.queue[slots.to(device)] = nn.functional.normalize(embeddings.detach(), dim=-1)
        self.families[slots] = families.detach().cpu()
        self.ptr = (self.ptr + n) % self.size


class EmbeddingLoss(nn.Module):
    def __init__(self, temperature=0.04, alpha=0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha       = alpha

    def infoNCE(self, queries, keys, queryFamilies, keyFamilies):
        """
        queries: [B, D]  — image or text embeddings from current batch
        keys:    [B+Q, D] — current batch + queue embeddings
        queryFamilies: [B]
        keyFamilies:   [B+Q]
        """
        queries = nn.functional.normalize(queries, dim=-1)
        keys    = nn.functional.normalize(keys,    dim=-1)

        logits = (queries @ keys.T) / self.temperature  # [B, B+Q]

        # Positive mask: same family OR diagonal (exact pair)
        familyMatch = (queryFamilies.unsqueeze(1) == keyFamilies.unsqueeze(0))  # [B, B+Q]
        # Diagonal of the batch portion is always the exact positive
        familyMatch[:, :queryFamilies.shape[0]].fill_diagonal_(True)

        targets = familyMatch.float()
        targets = targets / targets.sum(dim=1, keepdim=True)

        logSoftmax = torch.log_softmax(logits, dim=1)
        return -(targets * logSoftmax).sum(dim=1).mean()

    def forward(self, x, y, families=None, queue: MoCoQueue = None):
        B = x.shape[0]
        device = x.device

        if families is not None and queue is not None:
            queueEmb, queueFam = queue.get()
            queueEmb = queueEmb.to(device)
            queueFam = queueFam.to(device)

            # Keys = current batch + queue for both directions
            imageKeys = torch.cat([x, queueEmb], dim=0)
            allFamilies = torch.cat([families, queueFam], dim=0)

            loss1 = self.infoNCE(y, imageKeys, families, allFamilies)
            loss2 = self.infoNCE(x, y, families, families)

            # Enqueue current batch after computing loss
            queue.enqueue(x, families)

        elif families is not None:
            xNorm, yNorm = nn.functional.normalize(x, dim=-1), nn.functional.normalize(y, dim=-1)
            logits = (xNorm @ yNorm.T) / self.temperature
            familyMatch  = (families.unsqueeze(0) == families.unsqueeze(1))
            familyMatch.fill_diagonal_(True)
            targets = familyMatch.float()
            targets = targets / targets.sum(dim=1, keepdim=True)
            logSoftmax = torch.log_softmax(logits, dim=1)
            logSoftmax2 = torch.log_softmax(logits.T, dim=1)
            loss1 = -(targets * logSoftmax).sum(dim=1).mean()
            loss2 = -(targets.T * logSoftmax2).sum(dim=1).mean()

        else:
            xNorm, yNorm = nn.functional.normalize(x, dim=-1), nn.functional.normalize(y, dim=-1)
            logits = (xNorm @ yNorm.T) / self.temperature
            labels = torch.arange(B, device=device)
            loss1 = nn.functional.cross_entropy(logits,   labels)
            loss2 = nn.functional.cross_entropy(logits.T, labels)

        infoLoss = 0.5 * (loss1 + loss2)
        sigLoss  = 0.5 * (sigreg_weak_loss(x) + sigreg_weak_loss(y))
        return {"total": infoLoss + self.alpha * sigLoss, "info": infoLoss, "sig": sigLoss}


class Perplexity(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, y1, y2, names, queue):
        log = self.loss(y1, y2, names, queue)["info"]
        return torch.exp(log)
    

def recallAtK(x, y, families=None, k=10, queue: MoCoQueue = None):
    B = x.shape[0]
    device = x.device
    xNorm = nn.functional.normalize(x, dim=-1)
    yNorm = nn.functional.normalize(y, dim=-1)

    if queue is not None:
        queueEmb, queueFam = queue.get()
        queueEmb = nn.functional.normalize(queueEmb.to(device), dim=-1)
        queueFam = queueFam.to(device)
        keys = torch.cat([xNorm, queueEmb], dim=0)
        allFamilies = torch.cat([families, queueFam], dim=0)
    else:
        keys = xNorm
        allFamilies = families

    logits = yNorm @ keys.T  # [B, B+Q]

    if allFamilies is not None:
        positive = (families.unsqueeze(1) == allFamilies.unsqueeze(0))  # [B, B+Q]
    else:
        positive = torch.zeros(B, keys.shape[0], dtype=torch.bool, device=device)
        positive[:, :B].fill_diagonal_(True)

    topK = logits.topk(k, dim=1).indices
    hits = positive.gather(1, topK).any(dim=1).float()
    return hits.mean().item()