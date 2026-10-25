import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG6 layer config
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']


def get_activation(name='relu'):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"You gave wrong activation function i.e '{name}'. "
                         f"Supported ones are : ['relu', 'sigmoid', 'tanh', 'silu', 'gelu']")


def make_layers(cfg, activation='relu'):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                get_activation(activation),
            ]
            in_channels = v
    return nn.Sequential(*layers)


class VGG6(nn.Module):
    def __init__(self, num_classes=10, activation='relu'):
        super(VGG6, self).__init__()
        self.features   = make_layers(cfg_vgg6, activation=activation)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes=10, activation='relu'):
    return VGG6(num_classes=num_classes, activation=activation)
