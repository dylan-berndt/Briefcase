"""
Trains a multi-label tagging head on top of the pretrained ViT
(checkpoints/pretrain/latest, a base ViT). The backbone is frozen inside
ViTEmbedder; only its head learns to predict which style tags / adjectives a
font should be labeled with — the head's output width is the tag vocabulary size.

Saves the head to checkpoints/retrieval/{latest, <date>}/<experiment>/ and logs
to the "Font Retrieval" wandb project.
"""

import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import wandb

from utils import *


def itertoolsBetter(dataIter):
    while True:
        for batch in dataIter:
            yield batch


def computePosWeight(dataset, vocab):
    """
    pos_weight for BCEWithLogitsLoss: (#negatives / #positives) per tag, so rare
    tags aren't drowned out by the many fonts that lack them.
    """
    index = {tag: i for i, tag in enumerate(vocab)}
    counts = torch.zeros(len(vocab))
    for desc in dataset.descriptions.values():
        tags = set(desc.adjectives + list(desc.tags))
        for tag in tags:
            if tag in index:
                counts[index[tag]] += 1
    total = len(dataset.descriptions)
    return ((total - counts) / counts.clamp(min=1)).clamp(max=50.0)


@torch.no_grad()
def cheapMetrics(logits, targets, k=5):
    """F1 @0.5 plus sample-wise precision@k / recall@k. All on-device, every step."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    tp = (preds * targets).sum()
    precision = tp / (preds.sum() + 1e-8)
    recall = tp / (targets.sum() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    k = min(k, logits.shape[1])
    topk = probs.topk(k, dim=1).indices
    hits = targets.gather(1, topk)
    pAtK = (hits.sum(dim=1) / k).mean()
    rAtK = (hits.sum(dim=1) / targets.sum(dim=1).clamp(min=1)).mean()

    return {
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "pAtK": pAtK.item(),
        "rAtK": rAtK.item(),
    }


@torch.no_grad()
def meanAP(logits, targets):
    """Mean average precision across tags that have at least one positive in the batch."""
    from sklearn.metrics import average_precision_score

    probs = torch.sigmoid(logits).float().cpu().numpy()
    t = targets.float().cpu().numpy()
    aps = []
    for c in range(t.shape[1]):
        if t[:, c].sum() > 0:
            aps.append(average_precision_score(t[:, c], probs[:, c]))
    return float(np.mean(aps)) if aps else 0.0


def saveExperiment(model: ViTEmbedder, config, vocab, experimentName, start):
    stamp = start.strftime("%Y-%m-%d %H-%M")
    for path in [os.path.join("checkpoints", "retrieval", "latest"),
                 os.path.join("checkpoints", "retrieval", stamp)]:
        target = os.path.join(path, experimentName)
        os.makedirs(target, exist_ok=True)
        torch.save(model.head.state_dict(), os.path.join(target, "head.pt"))
        config.save(os.path.join(target, "config.json"))
        with open(os.path.join(target, "vocab.json"), "w+") as file:
            json.dump(vocab, file)


def trainModel(config, model: ViTEmbedder, dataset, posWeight, experimentName, start):
    model = model.to(device)
    print(f"Head has {sum(p.numel() for p in model.head.parameters())} trainable parameters "
          f"over {len(dataset.vocab)} tags")

    optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": config.learningRate},
                                #    {"params": model.model.parameters(), "lr": 1e-4}
                                   ], 
                                   lr=config.learningRate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=posWeight.to(device))

    print(f"{len(dataset)} total samples in dataset")

    train, test = CombinedQueryData.split(dataset, batchSize=config.batchSize)
    testIter = itertoolsBetter(test)

    run = None
    total = 0

    print("Beginning")

    try:
        for epoch in range(config.epochs):
            progress = 0
            for inputs, names, tokens, characters in train:
                model.train()
                optimizer.zero_grad()

                targets = tokens.float().to(device)
                logits = model(inputs.to(device))
                loss = criterion(logits, targets)

                trainLoss = loss.detach().item()
                loss.backward()
                optimizer.step()

                trainMetrics = cheapMetrics(logits.detach(), targets)

                with torch.no_grad():
                    model.eval()
                    inputs1, names1, tokens1, characters1 = next(testIter)
                    targets1 = tokens1.float().to(device)
                    logits1 = model(inputs1.to(device))
                    testLoss = criterion(logits1, targets1).item()
                    testMetrics = cheapMetrics(logits1, targets1)

                # mAP is the expensive one (per-tag sklearn) -> only every 25 steps
                payload = {
                    "Train Loss": trainLoss,
                    "Test Loss": testLoss,
                    "Train F1": trainMetrics["f1"],
                    "Test F1": testMetrics["f1"],
                    "Train Precision": trainMetrics["precision"],
                    "Test Precision": testMetrics["precision"],
                    "Train Recall": trainMetrics["recall"],
                    "Test Recall": testMetrics["recall"],
                    "Train Precision@5": trainMetrics["pAtK"],
                    "Test Precision@5": testMetrics["pAtK"],
                    "Train Recall@5": trainMetrics["rAtK"],
                    "Test Recall@5": testMetrics["rAtK"],
                }
                if total % 25 == 0:
                    payload["Train mAP"] = meanAP(logits.detach(), targets)
                    payload["Test mAP"] = meanAP(logits1, targets1)

                if run is None:
                    run = wandb.init(
                        entity="dylanberndt123-missouri-state-university",
                        project="Font Retrieval",
                        config=config.serialize(),
                    )

                run.log(payload, step=total)

                progress += 1
                total += 1
                print(f"\r{epoch + 1} | {progress}/{len(train)} | "
                      f"{(progress / len(train)) * 100:.2f}% | "
                      f"Train Loss: {trainLoss:.3f} | Test Loss: {testLoss:.3f} | "
                      f"Test F1: {testMetrics['f1']:.3f}", end="")

                if total % 1000 == 0:
                    saveExperiment(model, config, dataset.vocab, experimentName, start)

    except KeyboardInterrupt:
        pass

    print()
    saveExperiment(model, config, dataset.vocab, experimentName, start)
    return model


if __name__ == "__main__":
    config = Config().load(os.path.join("configs", "querying.json"))

    vit, imageConfig = ViT.load(os.path.join("checkpoints", "pretrain", "latest"))

    dataset = CombinedQueryData(imageConfig.dataset, training=True, multiClass=True)

    numTags = len(dataset.vocab)
    config.numTags = numTags
    config.task = "retrieval"
    config.backbone = os.path.join("checkpoints", "pretrain", "latest")

    # ViTEmbedder's head output width == number of tags, so it emits one logit per tag
    model = ViTEmbedder(vit, numTags)

    posWeight = computePosWeight(dataset, dataset.vocab)

    experimentName = "ViT tags"
    start = datetime.now()
    trainModel(config, model, dataset, posWeight, experimentName, start)
