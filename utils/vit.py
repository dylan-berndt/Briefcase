import torch
import torch.nn as nn
from .config import *
from .data import characters
import utils.unet

import os
import sys


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=config.embedDim,
            kernel_size=config.patchSize,
            stride=config.patchSize
        )

        numPatches = (config.imageSize // config.patchSize) ** 2
        self.positional = nn.Parameter(torch.zeros(1, numPatches, config.embedDim))

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional[:, :x.size(1), :]
        return x


class ViT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.patching = PatchEmbedding(config)
        self.clsToken = nn.Parameter(torch.randn(1, 1, config.embedDim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.embedDim, config.heads, config.embedDim * 3, batch_first=True),
            num_layers=config.layers
        )

        self.patches = config.imageSize // config.patchSize, config.imageSize // config.patchSize

        self.transpose = nn.ConvTranspose2d(config.embedDim, 1, kernel_size=config.patchSize, stride=config.patchSize)

        if "textProjection" not in config:
            self.classifier = nn.Sequential(
                nn.Linear(config.embedDim, config.embedDim),
                nn.ReLU(),
                nn.Linear(config.embedDim, len(characters))
            )
            self.outputType = "image"
        else:
            self.classifier = nn.Linear(config.embedDim, config.textProjection)
            self.outputType = "pooled"

        self.numLayers = config.layers * 2

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.patching(x)
        print(x.shape)
        x = torch.cat([self.clsToken.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.transformer(x)

        c = self.classifier(x[:, 0])
        if self.outputType == "pooled":
            return c
        
        x = x[:, 1:].view(x.shape[0], *self.patches, x.shape[-1])
        x = x.permute(0, 3, 1, 2)

        x = self.transpose(x)
        print(x.shape)

        return x, c

    
    def activations(self, x):
        raise NotImplementedError("Why am I even testing this model like this")