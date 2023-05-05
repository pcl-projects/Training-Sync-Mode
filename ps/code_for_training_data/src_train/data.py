#! /usr/bin/env python3


import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "/Users/zeyu/Documents/Seafile/CS Learning/Paper Projects/Sync & Async Para Updating/Experiment/Paper Exp./code_for_training_data/logs"


def get_wrk_comp_time_and_res(jid):
    rt_comp = [[] for _ in range(8)]
    rt_cpu = [[] for _ in range(8)]
    rt_bwin = [[] for _ in range(8)]
    rt_bwout = [[] for _ in range(8)]
    for wid in range(8):
        with open(f"{DATA_DIR}/job{jid}_worker{wid}.log") as f:
            line = f.readline()
            while line != "":
                if "comp." in line:
                    arr = line.rstrip("\n").split()
                    rt_comp[wid].append(float(arr[-1]))
                elif "res." in line:
                    arr = line.rstrip("\n").split()
                    rt_cpu[wid].append(float(arr[-3]))
                    rt_bwin[wid].append(float(arr[-2]))
                    rt_bwout[wid].append(float(arr[-1]))
                line = f.readline()
    for i in range(8):
        rt_comp[i] = rt_comp[i][1:]
        rt_cpu[i] = rt_cpu[i][1:]
        rt_bwin[i] = rt_bwin[i][1:]
        rt_bwout[i] = rt_bwout[i][1:]
    return rt_comp, rt_cpu, rt_bwin, rt_bwout


def get_wrk_comm_time(jid):
    rt = [[] for _ in range(8)]
    com_in = [[] for _ in range(8)]
    com_out = [[] for _ in range(8)]
    with open(f"{DATA_DIR}/job{jid}_ps0.log") as f:
        line = f.readline()
        while line != "":
            if "- wrk" in line:
                arr = line.rstrip("\n").split()
                wid = int(arr[-3])
                com_in[wid].append(float(arr[-2]))
                com_out[wid].append(float(arr[-1]))
            line = f.readline()
    for i in range(8):
        com_in[i] = com_in[i][1:]
        com_out[i] = com_out[i][2:]
    min_len = min([len(x) for x in com_in])
    min_len2 = min([len(x) for x in com_out])
    if min_len > min_len2:
        min_len = min_len2
    for i in range(8):
        for j in range(min_len):
            rt[i].append(com_in[i][j] + com_out[i][j])
    return rt


def get_wrk_iter_time(jid):
    rt = [[] for _ in range(8)]
    comp, _, _, _ = get_wrk_comp_time_and_res(jid)
    comm = get_wrk_comm_time(jid)
    for i in range(8):
        for j in range(len(comm[0])):
            rt[i].append(comm[i][j] + comp[i][j])
    return rt


itert = get_wrk_iter_time(1)
_, cpu, inbw, outbw = get_wrk_comp_time_and_res(1)
length = len(itert[0])
for i in range(8):
    cpu[i] = cpu[i][:length]
    inbw[i] = inbw[i][:length]
    outbw[i] = outbw[i][:length]


# p = (cpu, inbw, outbw, itert, "(CPU, BWin, BWout, IterTime)")
# with open("vgg13_data.pickle", "wb") as f:
#     pickle.dump(p, f)


class LstmRegrDataset(Dataset):
    def __init__(self, win_len=30, train=True):
        super().__init__()

        with open(
            "/Users/zeyu/Documents/Seafile/CS Learning/Paper Projects/Sync & Async Para Updating/Experiment/Paper Exp./code_for_training_data/data_train/vgg13_data.pickle",
            "rb",
        ) as f:
            cpu, inbw, outbw, itert, _ = pickle.load(f)

        cpu = torch.tensor(cpu, dtype=torch.float32)
        inbw = torch.tensor(inbw, dtype=torch.float32)
        outbw = torch.tensor(outbw, dtype=torch.float32)
        itert = torch.tensor(itert, dtype=torch.float32)
        cpu = cpu[:, :, None]
        inbw = inbw[:, :, None]
        outbw = outbw[:, :, None]
        itert = itert[:, :, None]
        self.data = torch.cat([cpu, inbw, outbw, itert], dim=2)

        self.train = train

        self.mean = torch.mean(self.data[:, : int(0.7 * len(self.data[0])), :], dim=(0, 1))
        self.std = torch.std(self.data[:, : int(0.7 * len(self.data[0])), :], dim=(0, 1))

        self.data = (self.data - self.mean) / self.std

        self.win_len = win_len

    def __len__(self):
        if self.train:
            return (int(0.7 * len(self.data[0])) - self.win_len) * len(self.data)
        else:
            return len(self.data[0]) - int(0.7 * len(self.data[0])) - self.win_len

    def __getitem__(self, index):
        if self.train:
            seqidx = int(index / len(self.data))
            x = self.data[index % len(self.data), seqidx : seqidx + self.win_len, :3]
            y = self.data[index % len(self.data), seqidx + 1 : seqidx + self.win_len + 1, :]
        else:
            shift = int(0.7 * len(self.data[0]))
            index = index + shift
            x = self.data[:, index : index + self.win_len, :3]
            y = self.data[:, index + 1 : index + self.win_len + 1, :]
        return x, y


if __name__ == "__main__":
    dataset = LstmRegrDataset()
    loader = DataLoader(dataset, shuffle=True, batch_size=20)
    for x, y in loader:
        print(x)
