#!/usr/bin/env python3
# Nexrad dataset class
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 11/13/2018

# load libs
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class NexradDataset(Dataset):
    """ NEXRAD dataset. """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the nexrad data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.categories = sorted(os.listdir(root))
        self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
        self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
        self.files = []
        
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            for f in filenames:
                if f.endswith('.csv'):
                    o = {}
                    o['radar_path'] = dirpath + '/' + f
                    o['category'] = self.cat2idx[dirpath[dirpath.rfind('/')+1:]]
                    self.files.append(o)
                    
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        radar_path = self.files[idx]['radar_path']
        category = self.files[idx]['category']
        radar = np.loadtxt(radar_path, delimiter=',')
        radar = radar.reshape((4, 60, 60))
        sample = {'radar': radar, 'category': category}
        
        if self.transform:
            sample = self.transform(sample)
			
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        radar, category = sample['radar'], sample['category']
        return {'radar': torch.from_numpy(radar),
                'category': torch.from_numpy(np.array(category, dtype=int))}

class Normalize(object):
    """Normalize a tensor radar with mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        for t, m, s in zip(sample['radar'], self.mean, self.std):
            t.sub_(m).div_(s)
        return sample
    
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
print('Sample radar shape: ', dset[0]['radar'].shape)


# dataloader

#dataloader = DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)
    
