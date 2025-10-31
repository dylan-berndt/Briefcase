import torch
import torch.nn as nn
from .config import *
from .data import characters
import utils.model

import os
import sys


class UNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        kernelSize = 3 if "kernelSize" not in config else config.kernelSize
        blockDepth = 3 if "blockDepth" not in config else config.blockDepth

        self.input = nn.Sequential(
            nn.Linear(1, config.filters),
            nn.ReLU()
        )

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for l in range(config.layers):
            filters = config.filters * (config.expansion ** l)
            layerConfig = Config(in_channels=filters, out_channels=filters * 2,
                                 kernel_size=kernelSize, block_depth=blockDepth)
            self.downs.append(Down(layerConfig))

            layerConfig.in_channels = layerConfig.in_channels * 4
            layerConfig.out_channels = layerConfig.out_channels // 2
            self.ups.append(Up(layerConfig))

        self.out = nn.Linear(config.filters, 1)

        width = config.filters * config.expansion ** config.layers
        self.classifier = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, len(characters))
        )

        self.numLayers = config.layers * 2
        self.outputType = "image"

    def forward(self, x):
        x = self.input(x)
        x = torch.permute(x, [0, 3, 2, 1])

        skips = []
        for l in range(self.config.layers):
            x, y = self.downs[l](x)
            skips.append(y)

        xMid = x.clone()

        for l, up in enumerate(reversed(self.ups)):
            y = skips[self.config.layers - l - 1]
            x, z = up(x, y)

        z = torch.permute(z, [0, 3, 2, 1])
        c = torch.mean(self.classifier(xMid.permute(0, 3, 2, 1)), dim=(1, 2))
        z = self.out(z)

        if self.outputType == "pooled":
            return c
        return z, c
    
    def activations(self, x):
        x = self.input(x)
        x = torch.permute(x, [0, 3, 2, 1])

        activations = []

        skips = []
        for l in range(self.config.layers):
            x, y = self.downs[l](x)
            skips.append(y)
            activations.append(y)

        for l, up in enumerate(reversed(self.ups)):
            y = skips[self.config.layers - l - 1]
            x, z = up(x, y)
            activations.append(z)

        return activations
    
    @staticmethod
    def load(path):
        modelPath = os.path.join(path, "checkpoint.pt")
        configPath = os.path.join(path, "config.json")

        loadedConfig = Config().load(configPath)
        loadedModel = UNet(loadedConfig.model)

        # Doofus. I saved the whole module, not just the state dict.
        sys.modules["model"] = utils.model
        sys.modules["config"] = utils.config
        loaded = torch.load(modelPath, weights_only=False)
        if hasattr(loaded, "state_dict"):
            loaded = loaded.state_dict()
        loadedModel.load_state_dict(loaded)
        loadedModel.eval()

        return loadedModel, loadedConfig


class Down(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = ConvBlock(config)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(config.out_channels)
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        y = self.conv(x)
        x = self.relu(self.bn(self.down(y)))

        y = self.drop(y)
        
        return x, y


class Up(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = ConvBlock(config)
        self.up = nn.Upsample(scale_factor=2)
        self.bn = nn.BatchNorm2d(config.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([x, y], dim=1)
        y = self.conv(x)
        x = self.relu(self.bn(y))

        return x, y


class ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        modules = nn.ModuleList([
            nn.Conv2d(config.in_channels, config.out_channels, config.kernel_size, padding="same"),
            nn.BatchNorm2d(config.out_channels),
            nn.ReLU()
        ])

        for layer in range(config.block_depth - 1):
            modules.extend([
                nn.Conv2d(config.out_channels, config.out_channels, config.kernel_size, padding="same"),
                nn.BatchNorm2d(config.out_channels),
                nn.ReLU()
            ])

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


if __name__ == "__main__":
    model = UNet(Config().load(os.path.join("configs", "config.json")).model)

    test = torch.randn(32, 40, 40, 1)

    outputs = model(test)

    print(outputs.shape)


