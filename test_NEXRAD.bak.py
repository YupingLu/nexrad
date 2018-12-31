#!/usr/bin/env python3
'''
Test script for NEXRAD
Different from test.py. This script is meant to test the raw four variable files.
Currently, this script only measures idx = [0, 60, 120, 180, 240, 300] idy = [0, 60] 
for each variable file.
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2018
Last Update: 12/31/2018
'''
# load libs
from __future__ import print_function
import sys

import pyart
from scipy.stats import mode
import numpy as np
import numpy.ma as ma

import os
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.nexraddataset import *
import models

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def test(args, model, device, test_loader, criterion):
    
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
            test_loss += loss.item() # sum up batch loss
            pred = outputs.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(labels).sum().item()
    
    # print average loss and accuracy
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss:\t'
          '{:.3f}\t'
          'Accuracy: {}/{}\t'
          '{:.3f}'.format(test_loss, correct, len(test_loader.dataset), acc))

    # np matrix to store the classification results
    #res = np.empty()

# Crop the file into 12 60*60 matrices
def datacrop(n0h, n0c, n0k, n0r, n0x):
    cnt = {
        30 : 'Ice Crystals', # Ice Crystals (IC) #
        40 : 'Dry Snow', # Dry Snow (DS) #
        60 : 'Rain', # Light and/or Moderate Rain (RA) #
        80 : 'Big Drops', # Big Drops (rain) (BD) #
    }

    idx = [0, 60, 120, 180, 240, 300]
    idy = [0, 60]

    # read data
    try:
        N0H = pyart.io.read(n0h)
    except:
        eprint("aa! N0H errors!")
        sys.exit(-1)
    try:
        N0C = pyart.io.read(n0c)
    except:
        eprint("aa! N0C errors!")
        sys.exit(-1)
    try:
        N0K = pyart.io.read(n0k)
    except:
        eprint("aa! N0K errors!")
        sys.exit(-1)
    try:
        N0R = pyart.io.read(n0r)
    except:
        eprint("aa! N0R errors!")
        sys.exit(-1)
    try:
        N0X = pyart.io.read(n0x)
    except:
        eprint("aa! N0X errors!")
        sys.exit(-1)
        
    # Check variable dims. If not match, stop.
    data_n0h = N0H.fields['radar_echo_classification']['data']
    data_n0c = N0C.fields['cross_correlation_ratio']['data']
    data_n0k = N0K.fields['specific_differential_phase']['data']
    data_n0r = N0R.fields['reflectivity']['data']
    data_n0x = N0X.fields['differential_reflectivity']['data']

    if data_n0h.shape != (360, 1200):
        eprint('Error dim: ' + n0h + '\n')
        sys.exit(-1)
    if data_n0c.shape != (360, 1200):
        eprint('Error dim: ' + n0c + '\n')
        sys.exit(-1)
    if data_n0k.shape != (360, 1200):
        eprint('Error dim: ' + n0k + '\n')
        sys.exit(-1)
    if data_n0r.shape != (360, 230):
        eprint('Error dim: ' + n0r + '\n')
        sys.exit(-1)
    if data_n0x.shape != (360, 1200):
        eprint('Error dim: ' + n0x + '\n')
        sys.exit(-1)

    # Extend n0r
    data_n0r_repeat = np.repeat(data_n0r, 5, axis=1)

    for j in range(len(idx)):
        for k in range(len(idy)):
            r1 = idx[j]
            c1 = idy[k]
            tmp_n0h = data_n0h[r1:r1+60, c1:c1+60]
            # mask 0, 10, 20, 140, 150
            # If the valid values of n0h is less then 6, abadon that entry.
            mx = ma.masked_values(tmp_n0h, 0.0) 
            mx = ma.masked_values(mx, 10.0) 
            mx = ma.masked_values(mx, 20.0)
            mx = ma.masked_values(mx, 140.0) 
            mx = ma.masked_values(mx, 150.0) 
            t_n0h = mx.compressed()
            unmask_size = len(t_n0h)
            if unmask_size < 12:
                eprint('Too few n0h: ' + n0h \
                                + ' ' + str(r1) + ' ' + str(c1) + '\n')
                continue
            # get the most frequent radar_echo_classification
            m = mode(t_n0h)
            res = m[0][0]
            if res < 6:
                eprint('Mode is small: ' + n0h \
                                + ' ' + str(r1) + ' ' + str(c1) + '\n')
                continue
            
            tmp_n0c = data_n0c[r1:r1+60, c1:c1+60]
            tmp_n0k = data_n0k[r1:r1+60, c1:c1+60]
            tmp_n0r = data_n0r_repeat[r1:r1+60, c1:c1+60]
            tmp_n0x = data_n0x[r1:r1+60, c1:c1+60]
            
            # Replace the missing values with mean values
            t_n0c = tmp_n0c.filled(tmp_n0c.mean())
            t_n0k = tmp_n0k.filled(tmp_n0k.mean())
            t_n0x = tmp_n0x.filled(tmp_n0x.mean())
            t_n0r = tmp_n0r.filled(tmp_n0r.mean())
            
            # Combine 4 2d array into 1 3d array
            f = open('/home/ylk/github/nexrad/tmp_test/'+cnt[res]+'/'+str(r1)+str(c1)+'.csv', 'wb')
            
            np.savetxt(f, t_n0c, delimiter=',')
            np.savetxt(f, t_n0k, delimiter=',')
            np.savetxt(f, t_n0r, delimiter=',')
            np.savetxt(f, t_n0x, delimiter=',')

            f.close()
            # Save results
            # res[r1:r1+60, c1:c1+60] = classification

#Visualize the classification results

def main():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='PyTorch NEXRAD Test')
    # Model options
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for test (default: 256)')
    #Device options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default='3', metavar='N',
                        help='id(s) for CUDA_VISIBLE_DEVICES (default: 3)')
    # Miscs
    parser.add_argument('--seed', type=int, default=20181212, metavar='S',
                        help='random seed (default: 20181212)')
    # Path to saved models
    parser.add_argument('--path', type=str, default='checkpoint/resnet18.pth.tar', metavar='PATH',
                        help='path to save models (default: checkpoint/resnet18.pth.tar)')

    args = parser.parse_args()
    
    # path to the raw data
    n0h = '/home/ylk/data/test_nexrad/processed/KOUN_SDUS84_N0HVNX_201801011620'
    n0c = '/home/ylk/data/test_nexrad/processed/KOUN_SDUS84_N0CVNX_201801011620'
    n0k = '/home/ylk/data/test_nexrad/processed/KOUN_SDUS84_N0KVNX_201801011620'
    n0r = '/home/ylk/data/test_nexrad/processed/KOUN_SDUS54_N0RVNX_201801011620'
    n0x = '/home/ylk/data/test_nexrad/processed/KOUN_SDUS84_N0XVNX_201801011620'

    datacrop(n0h, n0c, n0k, n0r, n0x)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        
    transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.7207, 0.0029, -1.6154, 0.5690],
                  std=[0.1592, 0.0570, 12.1113, 2.2363])
    ])
    
    testset = NexradDataset(root='/home/ylk/github/nexrad/tmp_test/', transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

    model = models.__dict__[args.arch](num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Load saved models.
    eprint("==> Loading model '{}'".format(args.arch))
    assert os.path.isfile(args.path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['model'])
    
    test(args, model, device, test_loader, criterion)
    
if __name__ == '__main__':
    main()
