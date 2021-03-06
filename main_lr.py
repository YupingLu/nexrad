#!/usr/bin/env python3
'''
Training script for NEXRAD with constant lr
Training set: 100,000 Validation set: 10,000 Test set: 10,000
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2018
Last Update: 12/19/2018
'''
# load libs
from __future__ import print_function
import sys

import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.nexraddataset import *
import models

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def train(args, model, device, train_loader, optimizer, criterion, epoch):

    model.train()
    train_loss = 0
    correct = 0
    acc = 0
    
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data['radar'].to(device), data['category'].to(device)
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # measure accuracy and record loss
        train_loss += loss.item() # sum up batch loss
        pred = outputs.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(labels).sum().item()
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
        '''
    # print average loss and accuracy
    train_loss /= len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    print('Train set: Average loss:\t'
          '{:.3f}\t'
          'Accuracy: {}/{}\t'
          '{:.3f}'.format(train_loss, correct, len(train_loader.dataset), acc))
    
def validation(args, model, device, validation_loader, criterion):
    
    model.eval()
    validation_loss = 0
    correct = 0
    acc = 0
    
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data['radar'].to(device), data['category'].to(device)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # measure accuracy and record loss   
            validation_loss += loss.item() # sum up batch loss
            pred = outputs.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(labels).sum().item()
    
    # print average loss and accuracy
    validation_loss /= len(validation_loader)
    acc = 100. * correct / len(validation_loader.dataset)
    print('Validation set: Average loss:\t'
          '{:.3f}\t'
          'Accuracy: {}/{}\t'
          '{:.3f}'.format(validation_loss, correct, len(validation_loader.dataset), acc))
    
    return acc

def main():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='PyTorch NEXRAD Training')
    # Model options
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                            ' (default: resnet18)')
    # Optimization options
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--validation-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for validation (default: 256)')
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                        help='resume epoch (default: 1')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='weight decay (default: 5e-4)')
    #Device options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default='3', metavar='N',
                        help='id(s) for CUDA_VISIBLE_DEVICES (default: 3)')
    # Miscs
    parser.add_argument('--seed', type=int, default=20181212, metavar='S',
                        help='random seed (default: 20181212)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='checkpoint', metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume from checkpoint')
    
    args = parser.parse_args()
    
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    train_transform = transforms.Compose([
        RandomCrop(padding=7),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Normalize(mean=[0.7207, 0.0029, -1.6154, 0.5690],
                  std=[0.1592, 0.0570, 12.1113, 2.2363])
    ])
    
    validation_transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.7207, 0.0029, -1.6154, 0.5690],
                  std=[0.1592, 0.0570, 12.1113, 2.2363])
    ])

    trainset = NexradDataset(root='/home/ylk/data/dataloader/train/', transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

    validationset = NexradDataset(root='/home/ylk/data/dataloader/validation/', transform=validation_transform)
    validation_loader = DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False, **kwargs)

    eprint("==> Building model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=4).to(device)

    best_acc = 0 # best validation accuracy
    start_epoch = args.start_epoch
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Load checkpoint.
    if args.checkpoint != 'checkpoint':
        cp = args.checkpoint
    else:
        cp = './checkpoint/' + args.arch + '.pth.tar'
    if args.resume:
        eprint('==> Resuming from checkpoint..')
        assert os.path.isfile(cp), 'Error: no checkpoint found!'
        checkpoint = torch.load(cp)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    for epoch in range(start_epoch, args.epochs + start_epoch):
        '''
        # check learning rate
        for param_group in optimizer.param_groups:
            eprint(param_group['lr'])
            break
        '''
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        acc = validation(args, model, device, validation_loader, criterion)
        # Save checkpoint.
        if acc > best_acc:
            eprint('Saving...{}'.format(acc))
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'model': model.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, cp)
            best_acc = acc
            
if __name__ == '__main__':
    main()