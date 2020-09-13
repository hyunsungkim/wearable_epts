import cv2
import torch
import argparse
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# Make custom dataset
class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx])
        # image = (image + 120.2075) / 20.1417
        label = int(self.labels[idx])
        return image, label
        
    def __len__(self):
        return self.images.shape[0]
    
# Define the network (fc)
# class CNN(nn.Module):
#     def __init__(self, n_classes=12):
#         super(CNN, self).__init__()
#         self.n_classes = n_classes
#         self.conv1 = nn.Conv2d(1, 5, kernel_size=5, padding=2)
#         self.pool1 = nn.AvgPool2d(kernel_size=4, stride=4)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(5, 10, kernel_size=5, padding=2)
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.relu2 = nn.ReLU()
#         self.fc1 = nn.Linear(240, n_classes)  # original -> no padding, 200 / M1 -> padding, 240

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.relu2(x)
#         x = x.view(-1, 240)
#         x = self.fc1(x)

#         return F.log_softmax(x, dim=1)

# Define the network (GAP)
class CNN(nn.Module):
    def __init__(self, n_classes=12):
        super(CNN, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--n_classes', type=int, required=True, help='number of classes in dataset')
    parser.add_argument('--n_samples', type=int, required=True, help='number of samples in one window')
    parser.add_argument('--distance', type=int, required=True, help='distance between successive two windows')

    args = parser.parse_args()

    n_samples = args.n_samples
    sample_distance = args.distance

    # Load activity image data
    images = np.load('./real-data/real_images_{}_{}_{}_0826_M1+M2+M3.npy'.format(n_samples, sample_distance, args.n_classes))
    labels = np.load('./real-data/real_labels_{}_{}_{}_0826_M1+M2+M3.npy'.format(n_samples, sample_distance, args.n_classes))

    # Data
    dataset = MyDataset(images=images, labels=labels)
    # mean, std = get_mean_and_std(dataset)
    # print('size: {}, mean: {}, std: {}'.format(len(dataset), mean, std))
    split_ratio = 1/6
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset=dataset, batch_size=128, sampler=train_sampler)
    test_loader = DataLoader(dataset=dataset, batch_size=128, sampler=test_sampler)

    # Parameters
    use_cuda = torch.cuda.is_available()
    n_epochs = args.n_epochs
    learning_rate = args.lr
    n_classes = args.n_classes

    # Model
    net = CNN(n_classes)

    if use_cuda:
        net.cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    # LR scheduler
    milestones = []
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)

    print("==> Training..")

    # Training
    for epoch in range(n_epochs):
        print('Epoch: {}'.format(epoch + 1))
        net.train()
        train_loss = 0
        correct_tr = 0
        total_tr = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            images = images.unsqueeze(1)
            optimizer.zero_grad()
            images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            loss_tr = criterion(outputs, labels)
            loss_tr.backward()
            optimizer.step()

            train_loss += loss_tr.item()
            _, predicted = torch.max(outputs.data, 1)
            total_tr += labels.size(0)
            correct_tr += predicted.eq(labels.data).cpu().sum()

        print('Train Loss: %.3f | Acc: %.3f%%'% (train_loss/128, 100.*correct_tr/total_tr))

        scheduler.step()

        # Test
        with torch.no_grad():
            net.eval()
            test_loss = 0
            correct_test = 0
            total_test = 0

            for batch_idx, (images, labels) in enumerate(test_loader):
                if use_cuda:
                    images, labels = images.cuda(), labels.cuda()
                images = images.unsqueeze(1)
                images, labels = Variable(images), Variable(labels)
                outputs = net(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels.data).cpu().sum()

            print('Test Loss: %.3f | Acc: %.3f%%\n' % (test_loss/128, 100.*correct_test/total_test))

    torch.save(net.state_dict(), './pretrained-real-data/MyModel.pth')
