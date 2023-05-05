#! /usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import shallow_reg_lstm as lstm

x = [[i, -i] for i in range(100)]
x = lstm.SequenceDataset(x, 20)
data_loader = DataLoader(x, batch_size=10, shuffle=True)

m = lstm.ShallowRegressionLSTM(num_features=2, hidden_units=32)
loss_fn_cpu = nn.MSELoss()
optimizer_cpu = torch.optim.Adam(m.parameters(), lr=5e-4)
for epc in range(1000):
    print(f"Epoch {epc}\n---------")
    lstm.train_model(data_loader, m, loss_fn_cpu, optimizer=optimizer_cpu)

a = lstm.predict(m, [[i, -i] for i in range(10, 29)])
print(a)
