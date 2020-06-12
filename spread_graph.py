#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# measure spread in activations, dropout, and identity

import glob
import json
import numpy as np

np.set_printoptions(linewidth=200)


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
identity_idx = None
for _i in range(len(activation_list)):
    if '.activation.' in activation_list[_i]:
        activation_idxs.append(_i)
    elif '.dropout.' in activation_list[_i]:
        dropout_idxs.append(_i)
    elif activation_list[_i] == 'torch.nn.modules.linear.Identity':
        identity_idx = _i

# spreads: a per-platform dict of sequences
activation_spreads = {}
dropout_spreads = {}

activation_mins = {}
activation_maxs = {}
activation_means = {}
dropout_mins = {}
dropout_maxs = {}
dropout_means = {}

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

print('activation spreads', activation_spreads)

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
ymax = 12
plt.axis(ymin=0,ymax=ymax,xmin=0,xmax=scale_length-1)
plt.yticks(range(ymax))
plt.grid(color='0.55', linestyle='dashed', lw=0.3)
plt.ylabel('spread')
plt.xlabel('number of instances, 10^n')
plt.title("Relative spread between function performances")
for prefix in d.keys():
    linestyle = next(linecycler)
    plt.plot(range(scale_length), activation_spreads[prefix], lw=1.5,
        color=next(main_colours), linestyle=linestyle, marker='.', ms=6, markerfacecolor='none')
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
# torch.nn.modules.linear.Identity


fig = plt.figure(figsize=(8,5))
plt.axis(ymin=0,ymax=7.5,xmin=-0.2,xmax=scale_length-0.5)
#plt.yscale('log')
plt.grid(color='0.55', linestyle='dashed', lw=0.3)

plt.ylabel('mean cost relative to identity activation')
plt.xlabel('number of instances, 10^n')

linecycler = cycle(lines)
main_colours = cycle([plt.cm.Paired(i) for i in np.linspace(0, 1, 12)])
#drop_colours =  cycle([plt.cm.autumn(i) for i in np.linspace(0, 1, 4)])

plt.title("Activation function time variation w.r.t identity function, over data scales")
for prefix in d.keys():
    print('prefix', prefix)
    identity_values = np.array(d[prefix][identity_idx]).reshape(9,1)
    print('identity values', identity_values)

    activation_values = list(d[prefix][_i] for _i in activation_idxs)
    activation_values = np.array(activation_values).T

    dropout_values = list(d[prefix][_i] for _i in dropout_idxs)
    dropout_values = np.array(dropout_values).T
    print('dropout values', dropout_values)

    #normalize
    print(activation_values.shape, identity_values.shape)
    activation_values = activation_values / identity_values
    dropout_values = dropout_values / identity_values

    activation_means = np.array(list(map(np.mean, activation_values)))
    activation_mins = list(map(np.min, activation_values))
    activation_maxs = list(map(np.max, activation_values))
    activation_stds = np.array(list(map(np.std, activation_values)))
    print('activation means', activation_means)
    print('activation stds', activation_stds)

    dropout_means = list(map(np.mean, dropout_values))
    dropout_mins = list(map(np.min, dropout_values))
    dropout_maxs = list(map(np.max, dropout_values))

    activation_colour = color=next(main_colours)
    #plt.plot(range(scale_length), activation_means, color=activation_colour)
    plt.errorbar(range(scale_length), activation_means, yerr=activation_stds,
        color=activation_colour, lw=1.5, elinewidth=1.5, capsize=8,
        linestyle=next(linecycler), fmt='x', ms=6)

# spreads: too big!
#    plt.fill_between(range(scale_length), activation_means+activation_stds,
#        activation_mins-activation_stds, color=activation_colour, alpha=0.1)


legend = [_l.replace('data/', '').replace('.npy', '').replace('_', ' ') for _l in d.keys()]
plt.legend(legend, loc='upper right',
    ncol=1, fontsize='small')

    #plt.plot(range(scale_length), dropout_means, color=next(drop_colours))

plt.show()
