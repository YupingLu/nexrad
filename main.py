#!/usr/bin/env python3
'''
Training script for NEXRAD
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2018
Last Update: 11/19/2018
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
import models

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    
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
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        
        train_loss += loss.item() # sum up batch loss
        pred = outputs.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(labels).sum().item()
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))

    # print average loss and accuracy
    train_loss /= len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset), acc))
    
    print(' * Loss@1 {loss.avg:.3f}'.format(loss=losses))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
def test(args, model, device, test_loader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['radar'].to(device), data['category'].to(device)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # measure accuracy and record loss
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            
            test_loss += loss.item() # sum up batch loss
            pred = outputs.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(labels).sum().item()
    
    # print average loss and accuracy
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    
    print(' * Loss@1 {loss.avg:.3f}'.format(loss=losses))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
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
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
                        help='number of epochs to train (default: 450)')
    parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                        help='resume epoch (default: 1')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='weight decay (default: 1e-4)')
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

    print("==> Building model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=4).to(device)

    best_acc = 0 # best test accuracy
    start_epoch = args.start_epoch
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Load checkpoint.
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile('./checkpoint/' + args.arch + '.pth.tar'), 'Error: no checkpoint found!'
        checkpoint = torch.load('./checkpoint/' + args.arch + '.pth.tar')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    for epoch in range(start_epoch, args.epochs + start_epoch):
        adjust_learning_rate(optimizer, epoch, args)
        '''
        # check learning rate
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            break
        '''
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        acc = test(args, model, device, test_loader, criterion)
        # Save checkpoint.
        if acc > best_acc:
            print('Saving..')
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'model': model.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict(),
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + args.arch + '.pth.tar')
            best_acc = acc

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
    lr = args.lr * (0.1 ** ((epoch-1) // 150))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
            
if __name__ == '__main__':
    main()