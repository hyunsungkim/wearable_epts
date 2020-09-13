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
from pathlib import Path
from time import time

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from model import CNN_m123 as CNN

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

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, required=False, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('--n_classes', type=int, required=False, default=6, help='number of classes in dataset')
    parser.add_argument('--n_samples', type=int, required=False, default=100, help='number of samples in one window')
    parser.add_argument('--output_dir', type=str, required=False, default='./models/', help='model output directory')
    parser.add_argument('--input_dir', type=str, required=False, default='./data/datasets_fin/', help='dataset directory')

    args = parser.parse_args()
    
    np.random.seed(42)

    n_samples = args.n_samples
    output_dir = args.output_dir
    
    train_images = np.load(Path(args.input_dir)/'train_images_imu_mix.npy')
    train_labels = np.load(Path(args.input_dir)/'train_labels_imu_mix.npy')
    test_images = np.load(Path(args.input_dir)/'test_images_imu_mix.npy')
    test_labels = np.load(Path(args.input_dir)/'test_labels_imu_mix.npy')
    
    train_dataset = MyDataset(images=train_images, labels=train_labels)
    test_dataset = MyDataset(images=test_images, labels=test_labels)
    # mean, std = get_mean_and_std(dataset)
    # print('size: {}, mean: {}, std: {}'.format(len(dataset), mean, std))

    train_indices, test_indices = list(range(len(train_images))), list(range(len(test_images)))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, sampler=test_sampler)
    
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
    
    torch.save(net.state_dict(), Path(output_dir)/f'MyModel.pth')

