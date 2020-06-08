#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# measure how long it takes to reach a given perf on mnist, varying activation

import time
import torch
import torchvision
from tqdm import tqdm


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

N, D_in, H, D_out = 128, 784, 1024, 10
accuracy_threshold = 0.9

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=N, shuffle=True)#, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=N, shuffle=False)#, num_workers=4)
dataiter = iter(train_loader)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.NLLLoss()

max_epochs = 50

times_taken = {}
accuracies = {}
for activation in activations:
    print('evaluating', activation)
    model = False
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            activation(),
            activation(),
            activation(),
            activation(),
            torch.nn.Linear(H, D_out),
            torch.nn.LogSoftmax(dim=1),
        )
    except:
        print('--failed')
        continue

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    time_taken = 0.0
    accuracy = 0.0
    e = 0
    try:
        while accuracy < accuracy_threshold and e < max_epochs:
            print('epoch', e, 'accuracy', accuracy, '/', accuracy_threshold)
            running_loss = 0
            for images, labels in tqdm(train_loader):
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
                start = time.perf_counter() # get counting
                # Training pass
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                #This is where the model learns by backpropagating
                loss.backward()
                #And optimizes its weights here
                optimizer.step()
                time_taken += time.perf_counter() - start # pause stopwatch

            correct_count, all_count = 0, 0
            for images,labels in test_loader:
              for i in range(len(labels)):
                img = images[i].view(1, 784)
                with torch.no_grad():
                    logps = model(img)

                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                  correct_count += 1
                all_count += 1

        #    print("Number Of Images Tested =", all_count)
            accuracy = correct_count/all_count
            e += 1
    except:
        print('-- training failed')
        continue

    print('activation', activation)
    print('final acc', accuracy)
    print('time', time_taken)
    times_taken[str(activation)] = time_taken
    accuracies[str(activation)] = accuracy

with open('mnist_targets.' + str(time.perf_counter()) + '.json', 'w') as f:
    f.write(json.dumps({'times':times_taken, 'accuracies':accuracies}))
