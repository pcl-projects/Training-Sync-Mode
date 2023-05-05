#! /usr/bin/env python3

# importing the libraries
import copy

# import matplotlib.pyplot as plt
import numpy as np
import torch

# importing the dataset
# from sklearn.datasets import load_breast_cancer
from torch import nn

# from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


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


class SVM(nn.Module):
    def __init__(self, input_shape):
        super(SVM, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_model(input_data, labels, batch_size, device, model=None):
    feature_num = len(input_data[0])

    if model is None:
        model = SVM(feature_num)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model = model.train()
    criterion = criterion.to(device)

    _, mean_lst, std_lst = standardize_data(input_data)
    train_dataset = RegDataset(input_data, labels)
    data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    for _ in range(3000):
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.cpu(), mean_lst, std_lst


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
    rt = float(result.to("cpu")[0][0])
    return rt
