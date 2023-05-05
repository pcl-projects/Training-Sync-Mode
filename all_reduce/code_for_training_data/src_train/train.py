#! /usr/bin/env python3

import data
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class LstmRegr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
        self.linear = nn.Linear(64, 3)

        self.fc1 = nn.Linear(3, 256)
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


def train_lstm_regr():
    m = LstmRegr()
    optimizer = optim.Adam(m.parameters())
    loss_fn = nn.MSELoss()
    # prepare data
    dataset = data.LstmRegrDataset()
    loader = DataLoader(dataset, shuffle=True, batch_size=20)
    dataset_test = data.LstmRegrDataset(train=False)
    loader_test = DataLoader(dataset_test, shuffle=True, batch_size=20)
    m.train()
    for i in range(1000):
        for x, y in loader:
            # print(x.shape)
            Y = m(x)
            loss = loss_fn(Y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 1 == 0:
            m.eval()
            total_count = 0
            correct_count = 0
            fp_total = 0
            fp_count = 0
            fn_total = 0
            fn_count = 0
            for x, y in loader_test:
                for idx in range(len(x)):
                    s_x = x[idx]
                    s_y = y[idx]
                    outputs = []
                    reals = []
                    for i in range(len(s_x)):
                        input = s_x[i : i + 1, :, :]
                        output = m(input)[0][-1][-1] * dataset.std[-1] + dataset.mean[-1]
                        outputs.append(float(output))
                        real = s_y[i][-1][-1] * dataset.std[-1] + dataset.mean[-1]
                        reals.append(float(real))
                    outputs.sort()
                    reals.sort()
                    pred_strg = False
                    real_strg = False
                    if (outputs[-1] - outputs[0]) / outputs[0] > 0.2:
                        pred_strg = True
                    if (reals[-1] - reals[0]) / reals[0] > 0.2:
                        real_strg = True
                    total_count += 1
                    if pred_strg is real_strg:
                        correct_count += 1
                    if real_strg:
                        fn_total += 1
                        if pred_strg is not real_strg:
                            fn_count += 1
                    else:
                        fp_total += 1
                        if pred_strg is real_strg:
                            fp_count += 1
            print(
                f"acc.: {correct_count/total_count:.3f} falsepositive: {fp_count/fp_total:.3f} falsenegative: {fn_count/fn_total:.3f}"
            )
            m.train()


if __name__ == "__main__":
    train_lstm_regr()
