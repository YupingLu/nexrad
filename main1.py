#!/usr/bin/env python3
'''
Training script for NEXRAD
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2018
Last Update: 11/15/2018
'''
# load libs
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.nexraddataset import *
#from models.vgg import *
from models.resnet import *

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data['radar'].to(device), data['category'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
        

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['radar'].to(device), data['category'].to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item() # sum up batch loss
            pred = outputs.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(labels).sum().item()
            
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # Optimization options
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                        help='resume epoch (default: 1')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='weight decay (default: 5e-1)')
    #Device options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default='3', metavar='N',
                        help='id(s) for CUDA_VISIBLE_DEVICES (default: 3)')
    # Miscs
    parser.add_argument('--seed', type=int, default=2018, metavar='S',
                        help='random seed (default: 2018)')
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
    
    data_transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.71055727, 0.00507260, -3.52237001, 0.26145971],
                  std=[0.16087105, 0.06199139, 13.40004994, 2.2214379])
    ])

    trainset = NexradDataset(root='/home/ylk/workspace/dataloader/train/', transform=data_transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

    testset = NexradDataset(root='/home/ylk/workspace/dataloader/test/', transform=data_transform)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    print('==> Building model..')
    #model = vgg16(num_classes=4).to(device)
    model = resnet18(num_classes=4).to(device)
    best_acc = 0 # best test accuracy
    start_epoch = args.start_epoch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/pth1.tar')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs + start_epoch):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        acc = test(args, model, device, test_loader, criterion)
        # Save checkpoint.
        if acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/pth1.tar')
            best_acc = acc

if __name__ == '__main__':
    main()