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
    activation_names = set()
    for run in d['pred_times']:
        activation_names = activation_names.union(set(run.keys()))
    activation_names = list(activation_names)

    # indices: device [cpu cuda], function, scale
    if absolute_timings is None:
        absolute_timings = np.zeros([2, len(activation_names), 9], dtype=np.float32)
    if per_inst_timings is None:
        per_inst_timings = np.zeros([2, len(activation_names), 9], dtype=np.float32)

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
        per_inst_timings[device_index, func_index, scale_index] = per_inst[activation_name][0]

        #print(activation_name, absolutes[activation_name], per_inst[activation_name])

    ## output device-instances-times.json
    #outfilename = 'analysis.' + '_'.join([d['device'],
    #    str(d['pred_items']), d['hardware_name'].replace(' ', '-')]) + '.json'
    #with open(outfilename, 'w') as outfile:
    #    json.dump(per_inst, outfile)
    #print('wrote', outfilename)

print(per_inst_timings)

cpu_df = pd.DataFrame(data=per_inst_timings[0,],
    index=[_n.split('.')[-1] for _n in activation_names], columns=range(9))
print(cpu_df)

import matplotlib.pyplot as plt
cpu_df_T = cpu_df.transpose()
cpu_df_T.plot.line(logy=True)
plt.show()

#import matplotlib.pyplot as plt
#plt.plot('log10 count of examples processed', 'per instance time', data=cpu_df)
#plt.plot('x', 'y', data=cpu_df)
#plt.legend()
#plt.show()
