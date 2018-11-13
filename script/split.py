#!/usr/bin/env python3
# Split datasets into training set and test set
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 11/13/2018

# 30 Ice\ Crystals  
# 40 Dry\ Snow  
# 60 Rain
# 80 Big\ Drops

# load libs
import pandas as pd
df = pd.read_csv('30.txt', header = None)  #

f_train = open("train_30.txt","w")  #
f_test = open("test_30.txt","w")    #
f_script = open("move_30.sh", "w")  #

for i in range(len(df.index)):    
    if (i+1) % 6 == 0:
        f_test.write(df.iloc[i,0] + '\n')
        f_script.write('mv train/Ice\ Crystals/' + df.iloc[i,0] + ' test/Ice\ Crystals/' + df.iloc[i,0] + '\n')  #
    else:
        f_train.write(df.iloc[i,0] + '\n')