#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# report on mnist power consumption to reach 0.9 accuracy

from collections import defaultdict
import glob
import json
import numpy as np

np.set_printoptions(linewidth=200)

learn_times = defaultdict(list)
low_acc = set()

for filename in glob.glob("data/mnist*json"):
    print('reading', filename)
    d = json.load(open(filename, 'r'))
    for func, perf in d['accuracies'].items():
        if perf < 0.9:
            low_acc.add(func)
    for func, time in d['times'].items():
        learn_times[func].append(time)

training_means = {f:np.mean(x) for f,x in learn_times.items()}
training_stdevs = {f:np.std(x) for f,x in learn_times.items()}
print('means', training_means)

training_means_sorted = sorted(training_means.items(), key=lambda x: x[1])

sorted_means = [v[1] for v in training_means_sorted]
sorted_names = [v[0] for v in training_means_sorted]
print('tms', training_means_sorted)

import matplotlib.pyplot as plt
from itertools import cycle
main_colours = cycle([plt.cm.cool(i) for i in np.linspace(0, 1, 20)])
drop_colours =  cycle([plt.cm.autumn(i) for i in np.linspace(0, 1, 6)])

colours = []
errors = []
legend = []
for _n in sorted_names:
    if _n in low_acc:
        colours.append('grey')
    else:
        if '.dropout' in _n:
            colours.append(next(drop_colours))
        elif '.activation.' in _n:
            colours.append(next(main_colours))
        elif '.identity' in _n:
            colours.append('white')
    errors.append(training_stdevs[_n])
    legend.append(_n.split('.')[-1].replace("'>", ''))

fig, ax = plt.subplots(figsize=(12,5))
plt.grid(color='0.55', linestyle='dashed', lw=0.3, axis='y')
plt.title('Time taken to reach 90% accuracy on MNIST data')
plt.xlabel('activation function')
plt.ylabel('seconds')
plt.bar(range(len(sorted_means)), sorted_means, color=colours)
plt.errorbar(range(len(sorted_means)), sorted_means, yerr=errors,
    color='black', lw=0, elinewidth=1, capsize=9)
ax.set_xticks(range(len(sorted_means)))
print('legend', legend)
ax.set_xticklabels(labels=legend, rotation=65)
fig.tight_layout()
fig.savefig('mnist_time.pdf')
fig.savefig('mnist_time.svg')
