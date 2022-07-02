"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2

from bcnet_dataset import BCNetDataset

def build_data_loader(dataset, items_per_batch):
    # sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, items_per_batch, drop_last=False)

    return torch.utils.data.DataLoader(dataset, num_workers=3, batch_sampler=batch_sampler, pin_memory=True)

def main():
    #dataset = BCNetDataset("D:\\Projects\\Research\\1-BCNet\\body_garment_dataset")
    dataset = BCNetDataset("/media/guomin/Works/Projects/Research/1-BCNet/body_garment_dataset")
    dataset.pre_process()

    #dataloader = build_data_loader(dataset=dataset, items_per_batch=3)

    #for iteration, item in enumerate(dataloader):
    #    print(iteration)
    #    print(item)


if __name__ == "__main__":
    main()
