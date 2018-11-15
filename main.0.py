#!/usr/bin/env python3

'''
Training script for NEXRAD
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2018
Last Update: 11/13/2018
'''

# load libs
#import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F  ####
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
#import models 
from datasets.nexraddataset import *

data_transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.71055727, 0.00507260, -3.52237001, 0.26145971],
                  std=[0.16087105, 0.06199139, 13.40004994, 2.2214379])
    ])    


dset = NexradDataset(root='/home/ylk/workspace/dataloader/train/', transform=data_transform)
print('######### Dataset class created #########')
print('Number of radar datasets: ', len(dset))
print('Number of categories: ', len(dset.categories))
print('Sample radar category 0: ', dset.idx2cat[dset[0]['category'].item()])
print('Sample radar category 1: ', dset.idx2cat[dset[12500]['category'].item()])
print('Sample radar category 2: ', dset.idx2cat[dset[25000]['category'].item()])
print('Sample radar category 3: ', dset.idx2cat[dset[37500]['category'].item()])
print('Sample radar shape: ', dset[0]['radar'].type())


dataloader = DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)

### Define your network below
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(60**2 * 4, 3600)
        self.fc2 = nn.Linear(3600, 4)

    def forward(self, x):
        x = x.view(-1, 60**2 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
print('######### Network created #########')
print('Architecture:\n', net)

### Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    examples = 0
    for i, data in enumerate(dataloader):
        # Get the inputs
        inputs, labels = data['radar'], data['category']

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        examples += 4
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / examples))

print('Finished Training')
