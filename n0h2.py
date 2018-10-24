#!/usr/bin/env python3
# Get the most frequent radar_echo_classification without 0, 10, 140, 150 and mode size > 20 
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 10/24/2018

# load libs
import pyart
from scipy.stats import mode
import numpy.ma as ma

cnt = {
    0 : 0, # Below Threshold (ND)
    10 : 0, # Biological (BI)
    20 : 0, # Anomalous Propagation/Group Clutter (GC)
    30 : 0, # Ice Crystals (IC)
    40 : 0, # Dry Snow (DS)
    50 : 0, # Wet Snow (WS)
    60 : 0, # Light and/or Moderate Rain (RA)
    70 : 0, # Heavy Rain (HR)
    80 : 0, # Big Drops (rain) (BD)
    90 : 0, # Graupel (GR)
    100 : 0, # Hail, possibly with rain (HA)
    140 : 0, # Unknown Classification (UK)
    150 : 0 # Range Folded (RH)
}

fw= open("label_2.txt","w")

with open('n0h.txt', 'r') as f:
    #counter = 0
    for line in f:
        #if counter == 100:
        #    break
        
        N0H = pyart.io.read('final/' + line.strip('\n'))
        x = N0H.fields['radar_echo_classification']['data']
        mx = ma.masked_values(x, 0.0) 
        mx = ma.masked_values(mx, 10.0) 
        mx = ma.masked_values(mx, 140.0) 
        mx = ma.masked_values(mx, 150.0) 
        data = mx.compressed()
        if len(data) != 0:
            d = mode(data)
            if d[1][0] > 20:
                res = d[0][0]
                cnt[res] += 1
                fw.write(str(int(res))+'\n')
                #print(res)
            else:
                fw.write('-1\n')
        else:
            fw.write('-1\n')
        
        #counter += 1
        
fw.close()
        
for key, value in cnt.items():
    print(key, "=>", value)