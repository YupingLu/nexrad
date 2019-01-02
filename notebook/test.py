# load the lib
import os
os.environ['PROJ_LIB'] = '/home/ylk/anaconda3/share/proj'
import pyart
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [35.0, 35.0]
import numpy as np
import warnings
warnings.filterwarnings('ignore')

N0H = pyart.io.read('/home/ylk/data/test_nexrad/processed/KOUN_SDUS84_N0HVNX_201801011620')
display = pyart.graph.RadarMapDisplay(N0H)
x = N0H.fields['radar_echo_classification']['data']

m = np.zeros_like(x)
m[:,120:]=1
y = np.ma.masked_array(x, m)
N0H.fields['radar_echo_classification']['data'] = y

fig = plt.figure(figsize=(6, 5))

# plot super resolution reflectivity
ax = fig.add_subplot(111)
display.plot('radar_echo_classification', 0, title='radar_echo_classification',
             colorbar_label='', ax=ax)
display.set_limits(xlim=(-40, 40), ylim=(-40, 40), ax=ax)
plt.show();

fig.savefig("test1.png", bbox_inches='tight')

m = np.zeros_like(x)
m[:,120:]=1
y = np.ma.masked_array(x, m)
y[:60,:120] = 3.3
y[60:120,:120] = 5.6
y[120:180,:120] = 2.1
y[180:240,:60] = 8
N0H.fields['radar_echo_classification']['data'] = y

fig = plt.figure(figsize=(6, 5))

# plot super resolution reflectivity
ax = fig.add_subplot(111)
display.plot('radar_echo_classification', 0, title='Test',
             colorbar_label='', ax=ax)
display.set_limits(xlim=(-40, 40), ylim=(-40, 40), ax=ax)
plt.show();

fig.savefig("test2.png", bbox_inches='tight')
