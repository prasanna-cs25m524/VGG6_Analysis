# model.py
# Responsible for: VGG6 architecture definition

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# VGG6 layer config: numbers = out_channels, 'M' = MaxPool
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ]
            in_channels = v
    return nn.Sequential(*layers)


class VGG6(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG6, self).__init__()
        self.features   = make_layers(cfg_vgg6)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # → (B, 128, 1, 1)
        x = torch.flatten(x, 1)               # → (B, 128)
        x = self.classifier(x)                # → (B, 10)
        return x


def get_model(num_classes=10):
    return VGG6(num_classes=num_classes)
