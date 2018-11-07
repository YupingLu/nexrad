#!/usr/bin/env python3
# Compute the mean and std of nexrad data
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 11/06/2018

# load libs
import numpy as np
import pandas as pd

root = '/home/ylk/workspace/dataloader/data/'
datasets = ['30', '40', '60', '80']

means = np.zeros([4])
total = []

for data in datasets:
    df = pd.read_csv(root + data + '.txt', header = None)
    
    #for i in range(len(df.index)):
    for i in range(2):
        d = np.loadtxt(root + data + '/' + df.iloc[i,0], delimiter=',')
        d = d.reshape((4, 60, 60))
        total.append(d)
        for j in range(4):
            pixel = d[j,:,:].ravel()
            means[j] += np.sum(pixel)

mean = []
stdevs = []
total = np.array(total)

for i in range(4):
    pixels = total[:,i,:,:].ravel()
    print(pixels.shape)
    mean.append(np.sum(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("means: {}".format(mean))
print("stdevs: {}".format(stdevs))