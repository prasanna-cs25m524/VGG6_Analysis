# dataset.py
# Responsible for: loading CIFAR-10 with transforms and returning dataloaders

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=64):
    # Training transforms: augmentation + normalize
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Test transforms: only normalize, no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10('./data', train=True,  transform=train_transform, download=True)
    test_data  = datasets.CIFAR10('./data', train=False, transform=test_transform,  download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
