import torch
import torch.nn as nn
from config import *
from data import characters

import os


class UNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.input = nn.Sequential(
            nn.Linear(1, config.filters),
            nn.ReLU()
        )

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for l in range(config.layers):
            filters = config.filters * (config.expansion ** l)
            layerConfig = Config(in_channels=filters, out_channels=filters * 2, kernel_size=3)
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

        return z, c


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

        self.module = nn.Sequential(
            nn.Conv2d(config.in_channels, config.out_channels, config.kernel_size, padding="same"),
            nn.BatchNorm2d(config.out_channels),
            nn.ReLU(),
            nn.Conv2d(config.out_channels, config.out_channels, config.kernel_size, padding="same"),
            nn.BatchNorm2d(config.out_channels),
            nn.ReLU(),
            nn.Conv2d(config.out_channels, config.out_channels, config.kernel_size, padding="same"),
            nn.BatchNorm2d(config.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.module(x)


if __name__ == "__main__":
    model = UNet(Config().load(os.path.join("configs", "config.json")).model)

    test = torch.randn(32, 40, 40, 1)

    outputs = model(test)

    print(outputs.shape)


