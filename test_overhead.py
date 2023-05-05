#! /usr/bin/env python3

import copy
import time

import numpy as np
import torch

import svm
from pgns import PGNS

svm_model = torch.load("./svm_model/svm_model.pth")
a = [0 for _ in range(13)]
b = [0 for _ in range(13)]
model_name = "vgg13"
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
oneshot = [0 for _ in range(10)]
i = 0
if model_name == "alexnet":
    i = 0
elif model_name == "resnet20":
    i = 1
elif model_name == "resnet56":
    i = 2
elif model_name == "vgg13":
    i = 3
elif model_name == "vgg16":
    i = 4
elif model_name == "densenet121":
    i = 5
elif model_name == "googlenet":
    i = 6
elif model_name == "mobilenet":
    i = 7
elif model_name == "lstm":
    i = 8
elif model_name == "transformer":
    i = 9
oneshot[i] = 1
oh_ml_time = 0

for _ in range(100):
    time_0 = time.time()
    input_data = []
    max_iter_time = 100
    avg_iter_time = np.average([1, 2, 3, 4, 5, 6, 7, 8, 9])
    straggling = (max_iter_time - avg_iter_time) / avg_iter_time
    input_data.append(straggling)

    input_data.extend(oneshot)
    input_data.append(0.05)
    input_data.append(3000)
    mode = svm.predict(svm_model, input_data, [0 for _ in range(13)], [0 for _ in range(13)], device)
    # self.set_sync_mode(mode)
    oh_ml_time += 1000 * (time.time() - time_0)
print("ml:", oh_ml_time)


# epoch_idx_sum = 0
# oh_hybrid_time = 0
# epoch = 3
# hybrid_pgns = PGNS("vgg16", int(8 * 128), "./gradient_validation/")
# next_iter_times = [7 for _ in range(8)]
# if epoch > 9:
#     epoch = 9

# for _ in range(100):
#     time_0 = time.time()
#     iter_times = copy.deepcopy(next_iter_times)
#     iter_times.sort()
#     k = 0
#     for i in range(1, 8):
#         t1 = hybrid_pgns.get_iter_num(int(i * 10), epoch) * iter_times[i - 1]
#         t2 = hybrid_pgns.get_iter_num(int((i + 1) * 10), epoch) * iter_times[i]
#         if t1 < t2:
#             k = i
#             break
#     if k == 0:
#         k == 8
#     hybrid_pgns.set_cur_bsz(int(k * 10))
#     hybrid_sync_k = k
#     oh_hybrid_time += 1000 * (time.time() - time_0)
# print("hybrid:", oh_hybrid_time)
import job_launcher as jl

jobs_data = [[1, 36, 89], [2, 66, 33], [3, 40, 86], [4, 88, 90], [5, 35, 58], [6, 77, 120], [7, 59, 32], [8, 33, 66]]
t = 0
for _ in range(100):
    time_0 = time.time()
    jl.ps_wrk_packing(jobs_data)
    t += 1000 * (time.time() - time_0)
print("pswrk:", t)


t = 0
jobs_data = {1: (36, 89), 2: [21, 33], 3: [150, 86], 0: [35, 58]}  # , 6: [77, 120]}  # , 7: [59, 32], 8: [33, 66]}
for _ in range(1):
    time_0 = time.time()
    jl.wrk_wrk_packing(jobs_data)
    t += 1000 * (time.time() - time_0)
print("wrkwrk:", t)
