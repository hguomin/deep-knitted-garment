# By Guomin Huang @2022.02.08
from __future__ import print_function, division
from email.mime import image
from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = "train"
validation = "val"

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
    "validation": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
}

data_folder = "datasets/hymenoptera_data"
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_folder, x), data_transforms["train"]) for x in [train, validation]
}
dataset_sizes = {
    x: len(image_datasets[x]) for x in [train, validation]
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in [train, validation]
}
labels = image_datasets[train].classes



