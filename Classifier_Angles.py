# Load libraries
import os

import numpy as np
import glob
import cv2
import os
from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import Adam, lr_scheduler
import torchvision
from torchvision import datasets, models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import pathlib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from PIL import Image

import time
import copy

# Define transforms for the training, validation, and testing sets
training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])


# Put the device to cuda or cpu :
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Path for training, validation and testing directory

data_dir1 = '/data/AIDatasets/AutomaticDataCleaning/Model_Angles/processed'

train_path1 = data_dir1 + '/train'
val_path1 = data_dir1 + '/val'
test_path1 = data_dir1 + '/test'

training_dataset1 = datasets.ImageFolder(train_path1, transform=training_transforms)
validation_dataset1 = datasets.ImageFolder(val_path1, transform=validation_transforms)
testing_dataset1 = datasets.ImageFolder(test_path1, transform=testing_transforms)

# Using the image datasets and the transforms, define the dataloaders :

train_loader1 = torch.utils.data.DataLoader(training_dataset1, batch_size=128, shuffle=True)
val_loader1 = torch.utils.data.DataLoader(validation_dataset1, batch_size=64)
test_loader1 = torch.utils.data.DataLoader(testing_dataset1, batch_size=64)


# Print the different classes :
root1=pathlib.Path(train_path1)
classes1=sorted([j.name.split('/')[-1] for j in root1.iterdir()])
print(classes1)

# Use transfer learning to get the model : 

Mymodel = models.resnet50(weights='ResNet50_Weights.DEFAULT')
# Freeze pretrained model parameters to avoid backpropogating through them

for parameter in Mymodel.parameters():
    parameter.requires_grad = False

Mymodel.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
Mymodel = Mymodel.to(device)

# Loss function and gradient descent

criterion = nn.BCEWithLogitsLoss()  # check if I need to change that
optimizer = optim.Adam(Mymodel.parameters(), lr=0.001, weight_decay=0.01)

def Sigmoid(output):

    Sigmoid_output = torch.max(torch.sigmoid(output), 1)
    for i in range(len(Sigmoid_output[0])):
        if Sigmoid_output[0][i] < 0.5:
            Sigmoid_output[1][i] = 2
    return  Sigmoid_output

def validation(model, validateloader, criterion):
    val_loss = 0
    accuracy = 0
    y_pred = []
    y_true = []
    Acc = []


    for images, labels in iter(validateloader):
        images, labels = images.to(device), labels.to(device)

        target_classes = F.one_hot(labels, num_classes=3)[:, 0:2]

        output = model.forward(images)
        Sigmoid_output = Sigmoid(output)

        val_loss += criterion(output, target_classes.float()).item()

        equality = (labels.data == Sigmoid_output[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        Acc.append(accuracy)

        outputN = (Sigmoid_output[1]).data.cpu().numpy()
        y_pred.extend(outputN)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    return val_loss, accuracy, y_pred, y_true

# Training and Validation loop :

EPOCHS = 100
epoch_count = []
train_loss_values = []
val_loss_values = []
train_accu = []
val_accuracy = []

for e in range(EPOCHS):

    Mymodel.train()
    running_loss = 0.0
    running_corrects = 0.0
    train_acc = 0

    for images, labels in iter(train_loader1):
        images, labels = images.to(device), labels.to(device)
        target_classes = F.one_hot(labels, num_classes=3)[:, 0:2] 

        optimizer.zero_grad()
        output = Mymodel.forward(images)
        Sigmoid_output = Sigmoid(output)

        loss = criterion(output, target_classes.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects = (labels.data == Sigmoid_output[1])
        train_acc += running_corrects.type(torch.FloatTensor).mean()

    Mymodel.eval()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        validation_loss, accuracy, y_pred, y_true = validation(Mymodel, val_loader1, criterion)

    epoch_count.append(e + 1)
    # losses
    train_loss_values.append(running_loss / len(train_loader1))
    val_loss_values.append(validation_loss / len(val_loader1))
    # Accuracies
    train_accu.append(train_acc / len(train_loader1))
    val_accuracy.append(accuracy / len(val_loader1))

    print("Epoch: {}/{}.. ".format(e + 1, EPOCHS),

          "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader1)),
          "Validation Loss: {:.3f}.. ".format(validation_loss / len(val_loader1)),
          "Training Accuracy: {:.3f}".format(train_acc / len(train_loader1)),
          "Validation Accuracy: {:.3f}".format(accuracy / len(val_loader1)))

    training_loss = 0
    Mymodel.train()

# Plot loss curves
plt.figure(1)
plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).cpu().numpy()), label="Training loss")
plt.plot(epoch_count, val_loss_values, label="Validation loss")
plt.title("Training and Validation loss curves")

plt.ylabel("Loss")
plt.xlabel("EPOCHS")
plt.legend()
plt.savefig('Loss_curve.png')

# Plot accuracy curves
plt.figure(2)
plt.plot(epoch_count, np.array(torch.tensor(train_accu).cpu().numpy()), label="Training accuracy")
plt.plot(epoch_count, np.array(torch.tensor(val_accuracy).cpu().numpy()), label="Validation accuracy")


plt.title("Validation and Training accuracy curves")

plt.ylabel("accuracy")
plt.xlabel("EPOCHS")
plt.legend()
plt.savefig('Accuracy_curve.png')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = [i for i in classes1])
cm_display.plot()
plt.show()
cm_display.figure_.savefig('confusion_matrix.png',dpi=100)


# Save the checkpoint of the model :
def save_checkpoint(model):

    model.class_to_idx = training_dataset1.class_to_idx

    checkpoint = {'arch': "Resnet50",
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                 }

    torch.save(checkpoint, 'checkpoint.pth')

save_checkpoint(Mymodel)
