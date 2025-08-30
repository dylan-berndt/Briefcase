import torch
import torch.nn as nn
from config import *

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
            # TODO: Uhhh the whole filters concatenating thing
            self.ups.append(Up(layerConfig))

        # TODO: Same sitch different line
        self.out = nn.Linear(config.filters, 1)

    def forward(self, x):
        for l in range(self.config.layers):
            pass


class Down(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = ConvBlock(config)
        self.down = nn.Conv2d(config.out_channels, config.out_channels, config.kernel_size, stride=2, padding="same")
        self.bn = nn.BatchNorm2d(config.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        x = self.relu(self.bn(self.down(y)))
        
        return x, y


class Up(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = ConvBlock(config)
        self.up = nn.Upsample()
        self.bn = nn.BatchNorm2d(config.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = torch.cat([x, y], dim=-1)
        x = self.conv(x)
        x = self.relu(self.bn(self.up(x)))

        return x


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
            nn.ReLU()
        )

    def forward(self, x):
        return self.module(x)


if __name__ == "__main__":
    model = UNet(Config().load(os.path.join("configs", "config.json")).model)

    test = torch.randn(32, 32, 32, 1)

    outputs = model(test)


