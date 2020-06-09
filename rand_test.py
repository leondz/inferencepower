#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# estimate power consumption of activation functions

import clize
import json
import math
import os
import random
import sys
import torch
import time
from tqdm import tqdm

# ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


torch.manual_seed(0)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1024, 1024, 16

def train_model(device, epochs=2000, activation=torch.nn.ReLU):

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        activation(),
        activation(),
        activation(),
        activation(),
        torch.nn.Linear(H, D_out),
    ).to(device)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(' -- training')
    for t in tqdm(range(epochs)):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def train_models(activations, epochs=2000, device_obj=None):
    models = {}
    train_times = {}

    if not os.path.isdir('model_cache'):
        os.mkdir('model_cache')

    # train models
    for func in activations:
        activation_name = str(func).split("'")[1]
        model_filename = 'model_cache/' + activation_name+'.model.pt'
        print(activation_name, end='')
        if not os.path.isfile(model_filename):
            try:
                start = time.perf_counter()
                m = train_model(device_obj, activation=func)
                elapsed = time.perf_counter() - start
                #print(" -- elapsed:", elapsed)
                train_times[activation_name] = elapsed
                torch.save(m, model_filename)
            except Exception as e:
                print(" XX couldn't train")
                pass
        else:
            print(' -- loaded')

        try:
            models[activation_name] = torch.load(model_filename)
            models[activation_name].to(device_obj)
            models[activation_name].eval()
        except FileNotFoundError:
            pass
    return models, train_times

def do_preds(models, test_values, pred_items, runs=3, impact_run_name=False):
    # track impact is either the run name, or False
    run_times = []
    model_names = list(models.keys())

    if impact_run_name:
        from experiment_impact_tracker.compute_tracker import ImpactTracker

    # eval loop
    for i in range(runs):
        pred_times = {}
        print('Run', i+1, 'of', runs)

        # shuffle function list each loop to reduce ordering effects
        random.shuffle(model_names)

        for func_name in tqdm(model_names):
            model = models[func_name]
            pred_item_count = 0
            #print(func_name)
            if impact_run_name: # are we trying to track carbon impact?
                with ImpactTracker(impact_run_name + '_' + func_name + '_' + str(i)) as tracker:
                    tracker.launch_impact_monitor()
                    start = time.perf_counter()
                    # use capped test_values size to conserve memory;
                    # go around until target # predictions made
                    while pred_item_count < pred_items:
                        predictions = model(test_values)
                        pred_item_count += test_values.shape[0]
                    elapsed = time.perf_counter() - start

            else:
                start = time.perf_counter()
                # use capped test_values size to conserve memory;
                # go around until target # predictions made
                while pred_item_count < pred_items:
                    predictions = model(test_values)
                    pred_item_count += test_values.shape[0]
                elapsed = time.perf_counter() - start

            #print('elapsed:', elapsed)
            #models[func_name] = m
            pred_times[func_name] = elapsed

        run_times.append(pred_times)
    return run_times

def measure_activations(*, scale=4, outprefix="test", device='cpu', runs=1,
    train_epochs = 2000, track_impact=False, test_size_cap = 100000):
    """Measures the time taken to train and infer using different activation
    functions

    :param scale: Perform predictions for 10^scale items
    :param outprefix: Prefix of the file to output to
    :param device: Torch device, e.g. cpu or cuda
    :param runs: How many runs to make
    :param train_epochs: Epoch count when training models
    :param track_impact: Try to load experiment_impact_tracker
    :param test_size_cap: Size of test chunk; fit as much in RAM as possible
    """
    saved_params=locals()

    run_name = outprefix + '_' + str(time.time())

    #activations = [torch.nn.ReLU]
    activations = [torch.nn.ReLU, torch.nn.ELU, torch.nn.Hardshrink,
        torch.nn.Hardtanh, torch.nn.LeakyReLU, torch.nn.LogSigmoid,
        torch.nn.MultiheadAttention, torch.nn.PReLU, torch.nn.ReLU,
        torch.nn.ReLU6, torch.nn.RReLU, torch.nn.SELU, torch.nn.CELU,
        torch.nn.GELU, torch.nn.Sigmoid, torch.nn.Softplus, torch.nn.Softshrink,
        torch.nn.Softsign, torch.nn.Tanh, torch.nn.Tanhshrink,
        torch.nn.Threshold, torch.nn.Softmin, torch.nn.Softmax,
        torch.nn.Softmax2d, torch.nn.LogSoftmax,
        torch.nn.AdaptiveLogSoftmaxWithLoss, torch.nn.Identity, torch.nn.Linear,
        torch.nn.Bilinear, torch.nn.Dropout, torch.nn.Dropout2d,
        torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.GroupNorm,
        torch.nn.SyncBatchNorm, torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm]

    # prefix for output files
    if not outprefix:
        outprefix = device.replace(':', '_')

    # set device
    device_obj = torch.device(device)
    print(device, ' - ', end='')
    hardware_name = 'unknown'
    if device == 'cpu':
        import cpuinfo
        hardware_name = cpuinfo.get_cpu_info()['brand']
    elif device == 'cuda':
        hardware_name = torch.cuda.get_device_name()
    print(hardware_name)

    # get the models, training if necessary
    models, train_times = train_models(activations, epochs=train_epochs, device_obj=device_obj)

    # build test data
    pred_items = int(math.pow(10, scale)) # how many predictions in total
    print('building random test data')
    if pred_items > test_size_cap:
        test_chunk_size = test_size_cap
    else:
        test_chunk_size = pred_items
    test_values = torch.randn(test_chunk_size, D_in, device=device_obj)

    # if we're tracking CO2, pass run name as the impact tracker directory
    impact_run_name = False
    if track_impact:
        impact_run_name = run_name

    # run inference
    pred_times = do_preds(models, test_values, pred_items, runs=runs,
        impact_run_name=impact_run_name)

    # save data
    experiment = {'train_epochs':train_epochs, 'pred_items':pred_items,
                    'train_times':train_times, 'pred_times':pred_times,
                    'device':device, 'test_chunk_size':test_chunk_size,
                    'runs':runs, 'outprefix':outprefix,
                    'parameters':saved_params, 'hardware_name':hardware_name,
                    'started_at':time.strftime('%c %Z')}
    outfilename = run_name + '.json'
    with open(outfilename, 'w') as outfile:
        outfile.write(json.dumps(experiment))

if __name__ == '__main__':
    clize.run(measure_activations)
