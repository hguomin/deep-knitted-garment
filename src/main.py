# By Guomin Huang @2022.02.08
from __future__ import print_function, division
from multiprocessing.dummy import freeze_support

from email.mime import image
from msilib.schema import Error
from pickletools import optimize
from pyexpat import model
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
import torch.backends.cudnn as cudnn


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(datasets, model, criterion, optimizer, scheduler, epochs = 25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {
        x: len(datasets[x]) for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ["train", "val"]
    }

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs-1))
        print("-" * 10)

        #Iterate over data
        for phase in ["train", "val"]:
            if  phase == "train":
                model.train() #Set model to training mode
            else:
                model.eval() #Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                #Zero the parameter gradients
                optimizer.zero_grad()

                #Forward
                #Track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    #Backward and optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step() #Calculate new parmeters using new gradients
                
                #Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == "train":
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() /dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(datasets, model, num_images=6):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ["train", "val"]
    }
    
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    plt.ion()

    train = "train"
    validation = "val"

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    }

    data_folder = 'D:\Projects\DZJ\deep-knitted-garment\datasets\hymenoptera_data'
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_folder, x), data_transforms[x]) for x in [train, validation]
    }
    dataset_sizes = {
        x: len(image_datasets[x]) for x in [train, validation]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in [train, validation]
    }
    labels = image_datasets[train].classes

    # Get a batch of training data
    #inputs, classes = next(iter(dataloaders["train"]))
    # Make a grid from batch
    #out = torchvision.utils.make_grid(inputs)
    #imshow(out, title=[labels[x] for x in classes])
    
    model_ft = models.resnet18(pretrained=True)
    # Here the size of each output sample is set to 2.
    model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(image_datasets, model_ft, criterion, optimizer_ft, exp_lr_scheduler)

    visualize_model(image_datasets, model_ft)
    
    print("Done")

if __name__ == "__main__":
    freeze_support()
    main()