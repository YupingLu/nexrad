#!/usr/bin/env python3
'''
Training script for NEXRAD
Only use portion of the input data for training.
Goal is to find the best hyper parameters for training
Training set: 10,000 Validation set: 10,000 Test set: 10,000
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2018
Last Update: 12/06/2018
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
from datasets.nexraddatasettest import *
import models