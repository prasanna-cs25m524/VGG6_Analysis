import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#These values are publicly available value and provided
#in sample code also
RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD  = (0.2023, 0.1994, 0.2010)

class Cutout(object):
    """
    Randomly mask out one or more square patches from an image tensor.
    Regularization technique to prevent lazy feature reliance.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with masked-out patch(es).
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        return img

def get_dataloaders(batch_size=64):
    # Training Transformation: augmentation + normalize
    train_transform = transforms.Compose([

        # add padding to images and then random crop
        # this will help for model not to learn about
        # only centered images
        transforms.RandomCrop(32, padding=4),

        # This is requred as the image can be left
        # or right faced
        transforms.RandomHorizontalFlip(),

        #slightly tilt the image
        transforms.RandomRotation(degrees=5),

        #tweak brightness, contrast, saturation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),

        # Not required no as I am not classifying any data
        # which supposed to be upside down
        # transforms.RandomVerticalFlip(),

        # convert PIL images to pyTorchTransform images
        transforms.ToTensor(),

        # pixel range sould be inn certain range based on
        # mean and SD
        transforms.Normalize(RGB_MEAN, RGB_STD),

        # randomly make a square patch and fill black
        Cutout(n_holes=1, length=16),
    ])

    # Test Transformation: only normalize no aug step
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD),
    ])

    #download the data from prof hinton website
    train_data = datasets.CIFAR10('./data', train=True,
                                  transform=train_transform,
                                  download=True)
    test_data  = datasets.CIFAR10('./data', train=False,
                                  transform=test_transform,
                                  download=True)

    #Engine that feeds data to model, shuffling, batching etc (:p) we are using GPU know thats why
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_data,
                              batch_size=batch_size,
                              shuffle=False, num_workers=0)

    return train_loader, test_loader
