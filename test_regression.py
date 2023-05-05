#! /usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import regression as reg

x = [[2, 3], [7, 8], [12, 13], [14, 9], [5, 7]]

y = [1, 4, 7, 10, 13]

model, mean_lst, std_lst = reg.train_model(x, y, 2)
a = reg.predict(model, [14, 9], mean_lst, std_lst)
print(a)
