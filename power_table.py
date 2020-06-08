#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# produce results tables of runs from rand_test.py (inference power)

import json
import numpy as np
import pandas as pd
import sys

json_filename = sys.argv[1]
d = json.load(open(json_filename, 'r'))

print('device:', d['device'])
print('hardware:', d['hardware_name'])
print('run from:', d['started_at'])
print('run size:', d['pred_items'])

# let's get a list of the functions in there
activation_names = set()
for run in d['pred_times']:
    activation_names = activation_names.union(set(run.keys()))
activation_names = list(activation_names)

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

    print(activation_name, absolutes[activation_name], per_inst[activation_name])

# output device-instances-times.json
outfilename = 'analysis.' + '_'.join([d['device'],
    str(d['pred_items']), d['hardware_name'].replace(' ', '-')]) + '.json'
with open(outfilename, 'w') as outfile:
    json.dump(per_inst, outfile)
