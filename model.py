import torch
import argparse
import torchvision
import torch.nn as nn
import numpy as npN
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
from tensorflow import keras

import os, sys

# Define the network (GAP)
class CNN_m123(nn.Module):
    def __init__(self, n_classes=12):
        super(CNN_m123, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        self.features = nn.Conv2d(10, n_classes, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.features(x)
        x = self.gap(x)
        x = x.view(-1, self.n_classes)

        return x

