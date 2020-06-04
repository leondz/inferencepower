#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from clize import run

import json
import math
import time
import torch
from tqdm import tqdm

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
        torch.nn.Linear(H, D_out),
    ).to(device)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algorithms. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in tqdm(range(epochs)):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        #if t % 100 == 99:
            #print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    return model

def measure_activations(*, scale=4, outprefix=None, device='cpu'):
    """Measures the time taken to train and infer using different activation
    functions

    :param scale: Perform predictions for 10^scale items
    :param outprefix: Prefix of the file to output to
    :param device: Torch device, e.g. cpu or cuda
    """
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

    models = {}
    train_times = {}
    pred_times = {}

    train_epochs = 2000

    if not outprefix:
        outprefix = device.replace(':', '_')
    device_obj = torch.device(device)
    print(device)

    # train models
    for func in activations:
        print(func)
        try:
            start = time.perf_counter()
            m = train_model(device_obj, activation=func)
            elapsed = time.perf_counter() - start
            print('elapsed:', elapsed)
            models[str(func)] = m
            train_times[str(func)] = elapsed
        except Exception as e:
            print("couldn't train")
            pass

    # run inference
    pred_items = int(math.pow(10, scale))
    print('building random data')
    test_values = torch.randn(pred_items, D_in, device=device_obj)

    for func_name, model in models.items():
        print(func_name)
        start = time.perf_counter()
        predictions = model(test_values)
        elapsed = time.perf_counter() - start
        print('elapsed:', elapsed)
        models[func_name] = m
        pred_times[func_name] = elapsed


    experiment = {'train_epochs':train_epochs, 'pred_items':pred_items,
                    'train_times':train_times, 'pred_times':pred_times,
                    'device':device}
    outfilename = outprefix + '_' + str(time.time()) + '.json'
    with open(outfilename, 'w') as outfile:
        outfile.write(json.dumps(experiment))

if __name__ == '__main__':
    run(measure_activations)
