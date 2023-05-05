#! /usr/bin/env python3


# An LSTM and regression combinded model for predicting workers' iteration times based on historical
# available CPU and bandwdith resource.


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class LstmRegr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=50, num_layers=2, batch_first=True)
        self.linear = nn.Linear(50, 2)

        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        hidden, _ = self.lstm(x)
        pred_res = self.linear(hidden)

        x = F.relu(self.fc1(pred_res))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        output = torch.cat([pred_res, x], dim=2)

        return output


class LstmRegrDataset(Dataset):
    def __init__(self, cpu_series, bw_series, itert_series, win_len=30):
        super().__init__()

        assert len(cpu_series) == len(bw_series) == len(itert_series)

        cpu = torch.tensor(np.array(cpu_series[:-1], dtype=np.float32)[:, np.newaxis], dtype=torch.float32)
        bw = torch.tensor(np.array(bw_series[:-1], dtype=np.float32)[:, np.newaxis], dtype=torch.float32)
        self.res = torch.cat((cpu, bw), dim=1)

        cpu = torch.tensor(np.array(cpu_series[1:], dtype=np.float32)[:, np.newaxis], dtype=torch.float32)
        bw = torch.tensor(np.array(bw_series[1:], dtype=np.float32)[:, np.newaxis], dtype=torch.float32)
        self.res_pred = torch.cat((cpu, bw), dim=1)

        self.itert = torch.tensor(np.array(itert_series[1:], dtype=np.float32)[:, np.newaxis], dtype=torch.float32)
        self.win_len = win_len

    def __len__(self):
        return len(self.cpu) - self.win_len

    def __getitem__(self, index):
        x = self.res[index, index + self.win_len]
        y1 = self.res_pred[index, index + self.win_len]
        y2 = self.itert[index, index + self.win_len]
        t = torch.cat([y1, y2], dim=1)
        return x, t


def train_lstm_regr(cpu, bw, itert, epoch):
    m = LstmRegr()
    optimizer = optim.Adam(m.parameters())
    loss_fn = nn.MSELoss()
    ds = LstmRegrDataset(cpu, bw, itert)
    data_loader = DataLoader(ds, shuffle=True, batch_size=20)
    m.train()
    for i in range(epoch):
        # m.train()
        for x, t in data_loader:
            y = m(x)
            loss = loss_fn(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
