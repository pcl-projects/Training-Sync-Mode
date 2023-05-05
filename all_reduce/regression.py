#! /usr/bin/env python3


import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Regression(nn.Module):
    def __init__(self, feature_num):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(feature_num, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.apply(_weight_init)  # 初始化参数

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def standardize_data(data):
    dim = len(data[0])
    x_num = len(data)
    trans_data = []
    for _ in range(dim):
        trans_data.append([])
    for sample_idx in range(x_num):
        for dim_idx in range(dim):
            trans_data[dim_idx].append(data[sample_idx][dim_idx])
    mean_lst = []
    std_lst = []
    for i in range(dim):
        values = np.array(trans_data[i], dtype=np.float32)
        mean_lst.append(values.mean())
        std_lst.append(values.std())
    rt_data = copy.deepcopy(data)
    for sample_idx in range(x_num):
        for dim_idx in range(dim):
            origin = data[sample_idx][dim_idx]
            if std_lst[dim_idx] == 0.0:
                rt_data[sample_idx][dim_idx] = 0.0
            else:
                rt_data[sample_idx][dim_idx] = (origin - mean_lst[dim_idx]) / std_lst[dim_idx]
    return rt_data, mean_lst, std_lst


class RegDataset(Dataset):
    def __init__(self, data, label):
        assert len(data) == len(label)
        data, _, _ = standardize_data(data)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)[:, None]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def train_model(input_data, labels, batch_size, device, model=None):
    feature_num = len(input_data[0])

    if model is None:
        model = Regression(feature_num)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00005, weight_decay=0)
    criterion = torch.nn.MSELoss()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model = model.train()
    criterion = criterion.to(device)

    _, mean_lst, std_lst = standardize_data(input_data)
    train_dataset = RegDataset(input_data, labels)
    data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    for _ in range(1000):
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.cpu(), mean_lst, std_lst


# def predict_wo_std(model, data, mean_lst, std_lst):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model = model.eval()
#     for i in range(len(data[0])):
#         v = data[0][i]
#         mean = mean_lst[i]
#         std = std_lst[i]
#         data[0][i] = (v - mean) / std
#     data = torch.tensor(data, dtype=torch.float32)
#     data = data.to(device)
#     result = model(data)
#     return float(result.to("cpu")[0][0])


def predict(model, data, mean_lst, std_lst, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()
    # stddata, mean_lst, std_list = standardize_data(stddata)
    dim = len(data)
    input_data = copy.deepcopy(data)
    for i in range(dim):
        if std_lst[i] == 0:
            input_data[i] = 0
        else:
            input_data[i] = (data[i] - mean_lst[i]) / std_lst[i]
    input_data = torch.tensor([input_data], dtype=torch.float32)
    input_data = input_data.to(device)
    # data = torch.tensor(data, dtype=torch.float32)
    # data = data.to(device)
    result = model(input_data)
    model = model.cpu()
    return float(result.to("cpu")[0][0])


# model = Regression()
# r = regression_pred(
#     model=model, data=[[1.0 for _ in range(32)]], mean_lst=[0 for _ in range(32)], std_lst=[1 for _ in range(32)]
# )
# print(r)
