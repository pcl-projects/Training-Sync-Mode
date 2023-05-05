#! /usr/bin/env python3


import argparse
import copy
import logging
import os
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from zeyu_utils import net as znet

from models import alexnet, densenet, googlenet, mobilenetv2, resnet3, vgg

model = resnet3.resnet56()
transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
training_dataset = datasets.CIFAR10(root="./training_data", train=True, download=True, transform=transform)
data_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.1)
device = torch.device("cuda:0")
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

a = []
for epoch_idx in range(1):
    for iter_idx, (data, target) in enumerate(data_loader):

        # self.model = self.model.to(device)
        data, target = data.to(device), target.to(device)
        model = model.to(device)
        model.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for k, p in model.named_parameters():
            print(p)
            break
        for k, p in model.named_parameters():
            a = model.state_dict()[k]
            print(a)
            print(a.type())
            model.state_dict()[k].copy_(a.cpu())
            break
        for k, p in model.named_parameters():
            print(p)
            break

        break
