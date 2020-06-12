#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# measure spread in activations, dropout, and identity

import glob
import json
import numpy as np

d = {}
for filename in glob.glob("data/*npy"):
    d[filename] = np.load(filename)
    print('loaded', filename)

activation_list = json.loads(open('dc_cpu_activation_list.json','r').read())
print('loaded activation list')

scale_length = 9

# get the activation and dropout names
activation_idxs = []
dropout_idxs = []
for _i in range(len(activation_list)):
    if '.activation.' in activation_list[_i]:
        activation_idxs.append(_i)
    elif '.dropout.' in activation_list[_i]:
        dropout_idxs.append(_i)

# spreads: a per-platform dict of sequences
activation_spreads = {}
dropout_spreads = {}


# go through each prefix
for prefix in d.keys():
# go for each scale point
    activation_spreads[prefix] = []
    dropout_spreads[prefix] = []
    for scale in range(scale_length):
        times = set()
        for _i in activation_idxs:
            times.add(d[prefix][_i][scale])
        scaled_spread = max(times) / min(times)
        activation_spreads[prefix].append(scaled_spread)

        times = set()
        for _i in dropout_idxs:
            times.add(d[prefix][_i][scale])
        scaled_spread = max(times) / min(times)

        dropout_spreads[prefix].append(scaled_spread)

print(activation_spreads)

# plot all the lines
import matplotlib.pyplot as plt
from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
main_colours = cycle([plt.cm.cool(i) for i in np.linspace(0, 1, 4)])
drop_colours =  cycle([plt.cm.autumn(i) for i in np.linspace(0, 1, 4)])


fig = plt.figure(figsize=(8,5))
#plt.yscale('log')
#plt.hlines([0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005], xmin, xmax, color='0.55', linestyle='dashed', lw=0.4)
plt.axis(ymin=0,ymax=12,xmin=0,xmax=scale_length-1)
plt.grid(color='0.55', linestyle='dashed', lw=0.3)
plt.ylabel('spread')
plt.xlabel('number of instances, 10^n')
plt.title("Relative spread between function performances")
for prefix in d.keys():
    linestyle = linestyle=next(linecycler)
    plt.plot(range(scale_length), activation_spreads[prefix], lw=1,
        color=next(main_colours), linestyle=linestyle, marker='.', ms=4.5, markerfacecolor='none')
    plt.plot(range(scale_length), dropout_spreads[prefix], lw=1,
        color=next(drop_colours), linestyle=linestyle)

legend = [_l.replace('data/', '').replace('.npy', '').replace('_', ' ') for _l in d.keys()]
doubled_legend = [val for val in legend for _ in (0, 1)]
for _j in range(len(d.keys())):
    doubled_legend[_j * 2] += ' activation'
    doubled_legend[1+ _j * 2] += ' dropout'

plt.legend(doubled_legend, loc='upper right',
    ncol=1, fontsize='small')

plt.show()

### second graph:
# with identity as 1.0, work out the relative scale and std dev of activations
# go through the scale
