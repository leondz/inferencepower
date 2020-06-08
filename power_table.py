#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# produce results tables of runs from rand_test.py (inference power)

import glob
import json
import numpy as np
import pandas as pd
import sys

np.set_printoptions(linewidth=200)

prefix = sys.argv[1]
print('prefix:', prefix)

absolute_timings, per_inst_timings = None, None
activation_names = None

filenames = glob.glob(prefix + '*.json')
for json_filename in filenames:
    d = json.load(open(json_filename, 'r'))
    print('read', json_filename)

    try:
        print('device:', d['device'])
        print('hardware:', d['hardware_name'])
        print('run from:', d['started_at'])
        print('run size:', d['pred_items'])
    except KeyError:
        print('-- metadata key missing, skipping')
        continue # if a key's not there, assume bad metadata


    # let's get a list of the functions in there
    if not activation_names:
        activation_names = set()
        for run in d['pred_times']:
            activation_names = activation_names.union(set(run.keys()))
        activation_names = list(activation_names)
        activation_names.sort()

    # indices: device [cpu cuda], function, scale
    if absolute_timings is None:
        absolute_timings = np.zeros([2, len(activation_names), 9], dtype=np.float32)
        absolute_std_dev = np.zeros([2, len(activation_names), 9], dtype=np.float32)
    if per_inst_timings is None:
        per_inst_timings = np.zeros([2, len(activation_names), 9], dtype=np.float32)
        per_inst_std_dev = np.zeros([2, len(activation_names), 9], dtype=np.float32)

    # let's get mean and variance per function
    timings = {}
    absolutes = {}
    per_inst = {}
    for activation_name in activation_names:
        timings[activation_name] = []
        for i in range(d['runs']):
            timings[activation_name].append(d['pred_times'][i][activation_name])
        mu, sigma = np.mean(timings[activation_name]), np.std(timings[activation_name])
        absolutes[activation_name] = (mu, sigma)
        per_inst[activation_name] = (mu/d['pred_items'], np.std([_j/d['pred_items']
            for _j in timings[activation_name]]) )

        func_index = activation_names.index(activation_name)
        scale_index = int(np.log10(d['pred_items']))
        device_index = 0 if d['device'] == 'cpu' else 1
        absolute_timings[device_index, func_index, scale_index] = absolutes[activation_name][0]
        absolute_std_dev[device_index, func_index, scale_index] = absolutes[activation_name][1]

        per_inst_timings[device_index, func_index, scale_index] = per_inst[activation_name][0]
        per_inst_std_dev[device_index, func_index, scale_index] = per_inst[activation_name][1]

        #print(activation_name, absolutes[activation_name], per_inst[activation_name])

    ## output device-instances-times.json
    #outfilename = 'analysis.' + '_'.join([d['device'],
    #    str(d['pred_items']), d['hardware_name'].replace(' ', '-')]) + '.json'
    #with open(outfilename, 'w') as outfile:
    #    json.dump(per_inst, outfile)
    #print('wrote', outfilename)

# per-instance timings with scale
print(per_inst_timings[0])
print('check: hash', hash(str(absolute_timings)), hash(str(per_inst_timings)))
print('check: sum ', sum(sum(per_inst_timings[0])))
print('check: row0', sum(per_inst_timings[0][0]))
print('check: col0', sum(per_inst_timings[0][:,0]))

import matplotlib.pyplot as plt
fig = plt.figure()

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
markers = '..vv^^++**'
markercycler = cycle(markers)

plt.yscale('log')
main_colours = cycle([plt.cm.cool(i) for i in np.linspace(0, 1, len(activation_names))])
drop_colours =  cycle([plt.cm.autumn(i) for i in np.linspace(0, 1, 6)])
for i in range(len(activation_names)):
    plot_colour = next(main_colours)
    if '.dropout.' in activation_names[i]:
        plot_colour = next(drop_colours)
    if '.linear.' in activation_names[i]:
        plot_colour = 'grey'
    plt.plot(range(9), per_inst_timings[0,i], color=plot_colour,
        linestyle=next(linecycler), lw=1, marker=next(markercycler), ms=6.5,
        markerfacecolor='none')
plt.legend([_n.split('.')[-1] for _n in activation_names], loc='upper right',
    ncol=2, fontsize='small')
plt.show()




if False:
    cpu_df = pd.DataFrame(data=per_inst_timings[0,],
    #    index=[_n.split('.')[-1] for _n in activation_names], columns=range(9))
        index=[activation_names], columns=range(9))
    print(cpu_df)

    import matplotlib.pyplot as plt

    cpu_df.T.plot.line(logy=True)
    plt.show()

    # absolute timing variances at scale.. 6
    # y = time; x = func, sorted by mean; item = box+whisker mean+stddev
    variance_scale = 6
