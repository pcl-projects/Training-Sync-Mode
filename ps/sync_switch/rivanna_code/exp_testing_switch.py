#! /usr/bin/env python3


# from keras.models import load_model
# import random
# import GPUtil as GPU
# import csv
import logging

# import os
# import pickle
# import subprocess
import threading
import time

# from multiprocessing import Process
from typing import List

# import jinja2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# from sklearn.cluster import KMeans
from zeyu_utils import net as znet

# from zeyu_utils import os as zos

# from kubernetes import client, config


# # Hyper Parameters
# BATCH_SIZE = 32
# LR = 0.01  # learning rate
# EPSILON = 0.9  # greedy policy
# GAMMA = 0.9  # reward discount
# EPOCH = 400
# TARGET_REPLACE_ITER = 100  # target update frequency
# MEMORY_CAPACITY = 30000
num_model_types = 10
num_task_types = 2
num_workers = 4
job_type_identity = np.identity(num_model_types)
task_type_identity = np.identity(num_task_types)

import os
import socket
import time
from _thread import *

# import numpy as np
#
# import pickle as cPickle
# import matplotlib.pyplot as plt
# import os
# os.environ['KERAS_BACKEND'] = 'theano'

# from threading import Thread as Process
# from multiprocessing import Process
# from multiprocessing import Manager
# from multiprocessing import Lock

# # import theano, theano.tensor as T
# import environment
# import job_distribution
# import pg_network
# # import pandas as pd


ServerSocket = socket.socket()
# host = 'udc-aw29-24a.hpc.virginia.edu'
host = "udc-ba26-24.hpc.virginia.edu"
port = 30000

file_path_prefix = "/home/qxc4fh/zeyu_workspace/sync_switch"

gpu_dev_list = [1, 2, 3]
gpu_dev_list_index = 0

list_server = [
    "udc-ba26-24.hpc.virginia.edu",
    "udc-ba26-25.hpc.virginia.edu",
    "udc-ba27-24.hpc.virginia.edu",
    "udc-ba27-23.hpc.virginia.edu",
    "udc-ba25-28.hpc.virginia.edu",
    "udc-ba25-27.hpc.virginia.edu",
]
# config.load_kube_config()

# model_name =['alexnet_v2', 'vgg_16', 'vgg_19', 'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'mobilenet_v1']

num_model_types = 10
num_task_types = 2
num_workers = 4
job_type_identity = np.identity(num_model_types)
task_type_identity = np.identity(num_task_types)
num_workers = 4

# global_port = 10000

# batch_size = 128

# ssgd_list = []
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})
# ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 32, "lr": 0.005})

ssgd_list = []
ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "resnet56", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "vgg13", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "vgg16", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "alexnet", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "mobilenet", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 128, "lr": 0.005})
ssgd_list.append({"model_name": "resnet20", "ss": "ssgd", "batch_size": 128, "lr": 0.005})


def threaded_client(connection):
    thread_starter = True
    # connection.send(str.encode('Welcome to the Server'))
    # while True:
    #     data = connection.recv(2048)
    #     reply = 'Server Says: ' + data.decode('utf-8')list_server = ['udc-ba25-27.hpc.virginia.edu', 'udc-ba25-28.hpc.virginia.edu', 'udc-ba27-23.hpc.virginia.edu', 'udc-ba27-24.hpc.virginia.edu', 'udc-aw37-37.hpc.virginia.edu', 'udc-aw37-33.hpc.virginia.edu', 'udc-aw37-33.hpc.virginia.edu', 'udc-an33-37.hpc.virginia.edu', 'udc-ba25-27.hpc.virginia.edu', 'udc-ba25-28.hpc.virginia.edu', 'udc-ba27-23.hpc.virginia.edu', 'udc-ba27-24.hpc.virginia.edu', 'udc-aw37-37.hpc.virginia.edu', 'udc-aw37-33.hpc.virginia.edu', 'udc-aw37-33.hpc.virginia.edu', 'udc-an33-37.hpc.virginia.edu', 'udc-ba25-27.hpc.virginia.edu', 'udc-ba25-28.hpc.virginia.edu', 'udc-ba27-23.hpc.virginia.edu', 'udc-ba27-24.hpc.virginia.edu', 'udc-aw37-37.hpc.virginia.edu', 'udc-aw37-33.hpc.virginia.edu', 'udc-aw37-33.hpc.virginia.edu', 'udc-an33-37.hpc.virginia.edu', 'udc-ba25-27.hpc.virginia.edu', 'udc-ba25-28.hpc.virginia.edu', 'udc-ba27-23.hpc.virginia.edu', 'udc-ba27-24.hpc.virginia.edu', 'udc-aw37-37.hpc.virginia.edu', 'udc-aw37-33.hpc.virginia.edu']
    #     if not data:
    #         break
    #     connection.sendall(str.encode(reply))
    # connection.close()


class ClassifierNet(torch.nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(in_dim, n_hidden)
        self.out_layer = torch.nn.Linear(n_hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.out_layer(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


classifier_net = ClassifierNet(71, 128, 2)
classifier_net.load_state_dict(torch.load(f"{file_path_prefix}/classifier_net.pt"))
classifier_net.eval()


class Logger(object):
    def __init__(self, job_name, file_dir, log_level=logging.INFO, mode="w"):
        self.logger = logging.getLogger(job_name)
        self.logger.setLevel(log_level)
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")

        self.fh = logging.FileHandler(filename=file_dir, mode=mode)
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)


class Scheduler(object):
    def __init__(self, max_jobs, time_list, job_list, model_list, slave_address, client_record):
        self.max_jobs = max_jobs
        self.job_list = job_list
        self.model_list = model_list
        self.slave_address = slave_address
        self.num_workers = 4
        # self.job_location = [np.zeros(self.num_workers + 1, dtype=np.int) for _ in range(len(self.job_list))]
        self.job_thread = [None for _ in range(len(self.job_list))]
        self.complete_flag = [False for _ in range(len(self.job_list))]
        self.entities = ["ps", "worker1", "worker2", "worker3", "worker4"]
        self.incoming_job = 0
        self.current_jobs = []
        self.completed_jobs = []
        self.waiting_jobs = []
        # self.batch_v1 = client.BatchV1Api()
        self.t0 = time.time()
        self.time_list = time_list
        self.port = 29600
        self.lock = threading.Lock()
        self.client_record = client_record
        self.global_port = 10000
        self.count_gpu = 0

    def switch(self, job_id, agent_connm: znet.SocketMsger, logger):
        time_interval = 25
        agent_connm.send(f"job{job_id}")

        # previous_state = None

        # if model_name == "resnet20":
        #     job_type = 0
        # elif model_name == "resnet56":
        #     job_type = 1
        # elif model_name == "vgg13":
        #     job_type = 2
        # elif model_name == "vgg16":
        #     job_type = 3
        # elif model_name == "resnet20":
        #     job_type = 4
        # elif model_name == "alexnet":
        #     job_type = 5
        # elif model_name == "resnet20":
        #     job_type = 6
        # elif model_name == "mobilenet":
        #     job_type = 7
        # elif model_name == "resnet20":
        #     job_type = 8
        # elif model_name == "resnet20":
        #     job_type = 9

        # if job_type < 8:
        #     dataset_size = 163.0
        #     job_task_type = 0
        #     batch_size = 128
        # else:
        #     dataset_size = 163.0
        #     job_task_type = 0
        #     batch_size = 128

        action = 0

        while True:

            if self.complete_flag[job_id] or agent_connm.closed is True:
                return

            time.sleep(time_interval)

            if self.complete_flag[job_id] or agent_connm.closed is True:
                return

            agent_connm.send("G")
            data = agent_connm.recv()

            if data is None:
                return

            all_worker_info = data[0]
            all_comm_time = data[1]
            ps_update_time = data[2]
            # resource_info = res_info[0]

            # res_arr = resource_info.split("|")
            # res_info_0 = res_arr[0].split()
            # res_info_1 = res_arr[1].split()
            # res_info_2 = res_arr[2].split()
            # res_info_3 = res_arr[3].split()
            # res_info_4 = res_arr[4].split()

            # job_type_ = job_type_identity[job_type]
            num_workers_ = num_workers * np.ones(1)
            # placement_ = placement.split(":")[1].split()
            # placement_ = np.array(list(map(float, placement_)))
            placement_ = [0, 0, 0, 0, 0]
            para_update_time = ps_update_time * np.ones(1)

            data_training_time = [
                all_worker_info["worker1"][2],
                all_worker_info["worker2"][2],
                all_worker_info["worker3"][2],
                all_worker_info["worker4"][2],
            ]
            data_training_time = np.array(list(map(float, data_training_time)))

            worker_to_ps_time = [
                all_comm_time["worker1"][0],
                all_comm_time["worker2"][0],
                all_comm_time["worker3"][0],
                all_comm_time["worker4"][0],
            ]
            worker_to_ps_time = np.array(list(map(float, worker_to_ps_time)))

            out_arr = np.add(data_training_time, worker_to_ps_time)
            min_t = min(out_arr)

            for item in out_arr:
                mult = item / min_t
                # print(mult)
                if mult > 1.5:
                    action = 1

            # ps_to_worker_time = [
            #     all_comm_time["worker1"][1],
            #     all_comm_time["worker2"][1],
            #     all_comm_time["worker3"][1],
            #     all_comm_time["worker4"][1],
            # ]
            # ps_to_worker_time = np.array(list(map(float, ps_to_worker_time)))
            # job_task_type_ = task_type_identity[job_task_type]
            # batch_size_ = batch_size * np.ones(1)
            # epoch = [
            #     all_worker_info["worker1"][0],
            #     all_worker_info["worker2"][0],
            #     all_worker_info["worker3"][0],
            #     all_worker_info["worker4"][0],
            # ]
            # epoch = np.array(list(map(float, epoch)))
            # iter = [
            #     all_worker_info["worker1"][1],
            #     all_worker_info["worker2"][1],
            #     all_worker_info["worker3"][1],
            #     all_worker_info["worker4"][1],
            # ]
            # iter = np.array(list(map(float, iter)))
            # dataset_size_ = dataset_size * np.ones(1)

            # server_cpu = [res_info_0[0], res_info_1[0], res_info_2[0], res_info_3[0], res_info_4[0]]
            # server_cpu = np.array(list(map(float, server_cpu)))
            # server_mem = [res_info_0[1], res_info_1[1], res_info_2[1], res_info_3[1], res_info_4[1]]
            # server_mem = np.array(list(map(float, server_mem)))
            # server_gpu_mem = [res_info_0[2], res_info_1[2], res_info_2[2], res_info_3[2], res_info_4[2]]
            # server_gpu_mem = np.array(list(map(float, server_gpu_mem)))
            # server_gpu = [res_info_0[3], res_info_1[3], res_info_2[3], res_info_3[3], res_info_4[3]]
            # server_gpu = np.array(list(map(float, server_gpu)))
            # server_band_in = [res_info_0[4], res_info_1[4], res_info_2[4], res_info_3[4], res_info_4[4]]
            # server_band_in = np.array(list(map(float, server_band_in)))
            # server_band_out = [res_info_0[5], res_info_1[5], res_info_2[5], res_info_3[5], res_info_4[5]]
            # server_band_out = np.array(list(map(float, server_band_out)))

            # ps_update_time = data[2] ##need to change later

            # job_type_ = job_type_identity[job_type]
            # num_workers_ = num_workers * np.ones(1)
            # placement_ = [0, 1, 2, 3, 4]
            # para_update_time = ps_update_time * np.ones(1)

            # data_training_time = get_data_training_time()

            # worker_to_ps_time = get_worker_to_ps_time()

            # ps_to_worker_time = get_ps_to_worker_time()

            # job_task_type_ = task_type_identity[job_task_type]

            # batch_size_ = batch_size * np.ones(1)

            # epoch = get_epoch()

            # iter = get_iter()

            # dataset_size_ = dataset_size * np.ones(1)

            # server_cpu = get_server_cpu()

            # server_mem = get_server_mem()
            # # server_mem = np.array(list(map(float, server_mem)))
            # server_gpu_mem =get_server_gpu_mem()
            # # server_gpu_mem = np.array(list(map(float, server_gpu_mem)))
            # server_gpu = get_server_gpu()
            # # server_gpu = np.array(list(map(float, server_gpu)))
            # server_band_in = get_server_band_in()
            # # server_band_in = np.array(list(map(float, server_band_in)))
            # server_band_out = get_server_band_out()
            # # server_band_out = np.array(list(map(float, server_band_out)))

            # input = np.concatenate(
            #     [
            #         job_type_,
            #         num_workers_,
            #         placement_,
            #         para_update_time,
            #         data_training_time,
            #         worker_to_ps_time,
            #         ps_to_worker_time,
            #         job_task_type_,
            #         batch_size_,
            #         epoch,
            #         iter,
            #         dataset_size_,
            #         server_cpu,
            #         server_mem,
            #         server_gpu_mem,
            #         server_gpu,
            #         server_band_in,
            #         server_band_out,
            #     ]
            # )

            # input = torch.tensor([input], dtype=torch.float32)
            # with torch.no_grad():
            #     output = classifier_net(input)
            # action = int(torch.max(output, 1)[1][0])

            agent_connm.send(f"{action}")

            logger.info(f"=== sync_mode: {action} ===")

            time.sleep(200)

            # data_action = []
            # data_action.append(action)
            # # data_iteration.append(duration)

            # data_action_csv = []
            # data_action_csv.append(data_action)

            # import csv
            # file_action = 'job_id_action.csv'
            # with open(file_action, 'a', newline ='') as file:
            #     wr = csv.writer(file)
            #     wr.writerows(data_action_csv)

            if self.complete_flag[job_id] or agent_connm.closed is True:
                return

    def make_decision(self, job_id, model_name, placement, agent_connm: znet.SocketMsger, res_info: List[str], logger):

        time_interval = 25
        agent_connm.send(f"job{job_id}")

        # previous_state = None

        if model_name == "resnet20":
            job_type = 0
        elif model_name == "resnet56":
            job_type = 1
        elif model_name == "vgg13":
            job_type = 2
        elif model_name == "vgg16":
            job_type = 3
        elif model_name == "resnet20":
            job_type = 4
        elif model_name == "alexnet":
            job_type = 5
        elif model_name == "resnet20":
            job_type = 6
        elif model_name == "mobilenet":
            job_type = 7
        elif model_name == "resnet20":
            job_type = 8
        elif model_name == "resnet20":
            job_type = 9

        if job_type < 8:
            dataset_size = 163.0
            job_task_type = 0
            batch_size = 128
        else:
            dataset_size = 163.0
            job_task_type = 0
            batch_size = 128

        while True:
            if self.complete_flag[job_id] or agent_connm.closed is True:
                return

            time.sleep(time_interval)

            if self.complete_flag[job_id] or agent_connm.closed is True:
                return

            agent_connm.send("G")
            data = agent_connm.recv()

            if data is None:
                return

            all_worker_info = data[0]
            all_comm_time = data[1]
            ps_update_time = data[2]
            resource_info = res_info[0]

            res_arr = resource_info.split("|")
            res_info_0 = res_arr[0].split()
            res_info_1 = res_arr[1].split()
            res_info_2 = res_arr[2].split()
            res_info_3 = res_arr[3].split()
            res_info_4 = res_arr[4].split()

            job_type_ = job_type_identity[job_type]
            num_workers_ = num_workers * np.ones(1)
            # placement_ = placement.split(":")[1].split()
            # placement_ = np.array(list(map(float, placement_)))
            placement_ = [0, 0, 0, 0, 0]
            para_update_time = ps_update_time * np.ones(1)
            data_training_time = [
                all_worker_info["worker1"][2],
                all_worker_info["worker2"][2],
                all_worker_info["worker3"][2],
                all_worker_info["worker4"][2],
            ]
            data_training_time = np.array(list(map(float, data_training_time)))
            worker_to_ps_time = [
                all_comm_time["worker1"][0],
                all_comm_time["worker2"][0],
                all_comm_time["worker3"][0],
                all_comm_time["worker4"][0],
            ]
            worker_to_ps_time = np.array(list(map(float, worker_to_ps_time)))
            ps_to_worker_time = [
                all_comm_time["worker1"][1],
                all_comm_time["worker2"][1],
                all_comm_time["worker3"][1],
                all_comm_time["worker4"][1],
            ]
            ps_to_worker_time = np.array(list(map(float, ps_to_worker_time)))
            job_task_type_ = task_type_identity[job_task_type]
            batch_size_ = batch_size * np.ones(1)
            epoch = [
                all_worker_info["worker1"][0],
                all_worker_info["worker2"][0],
                all_worker_info["worker3"][0],
                all_worker_info["worker4"][0],
            ]
            epoch = np.array(list(map(float, epoch)))
            iter = [
                all_worker_info["worker1"][1],
                all_worker_info["worker2"][1],
                all_worker_info["worker3"][1],
                all_worker_info["worker4"][1],
            ]
            iter = np.array(list(map(float, iter)))
            dataset_size_ = dataset_size * np.ones(1)

            server_cpu = [res_info_0[0], res_info_1[0], res_info_2[0], res_info_3[0], res_info_4[0]]
            server_cpu = np.array(list(map(float, server_cpu)))
            server_mem = [res_info_0[1], res_info_1[1], res_info_2[1], res_info_3[1], res_info_4[1]]
            server_mem = np.array(list(map(float, server_mem)))
            server_gpu_mem = [res_info_0[2], res_info_1[2], res_info_2[2], res_info_3[2], res_info_4[2]]
            server_gpu_mem = np.array(list(map(float, server_gpu_mem)))
            server_gpu = [res_info_0[3], res_info_1[3], res_info_2[3], res_info_3[3], res_info_4[3]]
            server_gpu = np.array(list(map(float, server_gpu)))
            server_band_in = [res_info_0[4], res_info_1[4], res_info_2[4], res_info_3[4], res_info_4[4]]
            server_band_in = np.array(list(map(float, server_band_in)))
            server_band_out = [res_info_0[5], res_info_1[5], res_info_2[5], res_info_3[5], res_info_4[5]]
            server_band_out = np.array(list(map(float, server_band_out)))
            #
            # ps_update_time = data[2] ##need to change later

            # job_type_ = job_type_identity[job_type]
            # num_workers_ = num_workers * np.ones(1)
            # placement_ = [0, 1, 2, 3, 4]
            # para_update_time = ps_update_time * np.ones(1)

            # data_training_time = get_data_training_time()

            # worker_to_ps_time = get_worker_to_ps_time()

            # ps_to_worker_time = get_ps_to_worker_time()

            # job_task_type_ = task_type_identity[job_task_type]

            # batch_size_ = batch_size * np.ones(1)

            # epoch = get_epoch()

            # iter = get_iter()

            # dataset_size_ = dataset_size * np.ones(1)

            # server_cpu = get_server_cpu()

            # server_mem = get_server_mem()
            # # server_mem = np.array(list(map(float, server_mem)))
            # server_gpu_mem =get_server_gpu_mem()
            # # server_gpu_mem = np.array(list(map(float, server_gpu_mem)))
            # server_gpu = get_server_gpu()
            # # server_gpu = np.array(list(map(float, server_gpu)))
            # server_band_in = get_server_band_in()
            # # server_band_in = np.array(list(map(float, server_band_in)))
            # server_band_out = get_server_band_out()
            # # server_band_out = np.array(list(map(float, server_band_out)))

            input = np.concatenate(
                [
                    job_type_,
                    num_workers_,
                    placement_,
                    para_update_time,
                    data_training_time,
                    worker_to_ps_time,
                    ps_to_worker_time,
                    job_task_type_,
                    batch_size_,
                    epoch,
                    iter,
                    dataset_size_,
                    server_cpu,
                    server_mem,
                    server_gpu_mem,
                    server_gpu,
                    server_band_in,
                    server_band_out,
                ]
            )

            input = torch.tensor([input], dtype=torch.float32)
            with torch.no_grad():
                output = classifier_net(input)
            action = int(torch.max(output, 1)[1][0])

            agent_connm.send(f"{action}")

            logger.info(f"=== sync_mode: {action} ===")

            # data_action = []
            # data_action.append(action)
            # # data_iteration.append(duration)

            # data_action_csv = []
            # data_action_csv.append(data_action)

            # import csv
            # file_action = 'job_id_action.csv'
            # with open(file_action, 'a', newline ='') as file:
            #     wr = csv.writer(file)
            #     wr.writerows(data_action_csv)

            if self.complete_flag[job_id] or agent_connm.closed is True:
                return

    def get_data_training_time():
        file_read_comp = "computation_time.csv"
        df_comp = pd.read_csv(file_read_comp, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        comp_time_all = pd.DataFrame(df_comp.values[:, 1])
        # print(itn_time_all)

        comp_time_all = comp_time_all.to_numpy()

        workers_comp = comp_time_all[len(comp_time_all) - 4 : len(comp_time_all)]

        return workers_comp

    def get_worker_to_ps_time():
        file_read_comm = "communication_time.csv"
        df_comm = pd.read_csv(file_read_comm, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        comm_time_all = pd.DataFrame(df_comm.values[:, 1])
        # print(itn_time_all)

        comm_time_all = comm_time_all.to_numpy()

        workers_comm = comm_time_all[len(comm_time_all) - 4 : len(comm_time_all)]

        return workers_comm

    def get_ps_to_worker_time():
        file_read_comm = "ps_to_w_time.csv"
        df_comm = pd.read_csv(file_read_comm, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        comm_time_all = pd.DataFrame(df_comm.values[:, 1])
        # print(itn_time_all)

        comm_time_all = comm_time_all.to_numpy()

        workers_comm = comm_time_all[len(comm_time_all) - 4 : len(comm_time_all)]

        return workers_comm

    def get_epoch():
        file_read_e = "w_epoch.csv"
        df_e = pd.read_csv(file_read_e, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        e_all = pd.DataFrame(df_e.values[:, 1])
        # print(itn_time_all)

        e_all = e_all.to_numpy()

        workers_e = e_all[len(e_all) - 4 : len(e_all)]

        return workers_e

    def get_iter():
        file_read_itn = "w_itn.csv"
        df_itn = pd.read_csv(file_read_itn, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        itn_all = pd.DataFrame(df_itn.values[:, 1])
        # print(itn_time_all)

        itn_all = itn_all.to_numpy()

        workers_itn = itn_all[len(itn_all) - 4 : len(itn_all)]

        return workers_itn

    def get_server_cpu():
        file_read_s_cpu = "server_cpu.csv"
        df_s_cpu = pd.read_csv(file_read_s_cpu, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        s_cpu_all = pd.DataFrame(df_s_cpu.values[:, 1])
        # print(itn_time_all)

        s_cpu_all = s_cpu_all.to_numpy()

        s_cpu = s_cpu_all[len(s_cpu_all) - 5 : len(s_cpu_all)]

        return s_cpu

    def get_server_mem():
        file_read_s_mem = "server_mem.csv"
        df_s_mem = pd.read_csv(file_read_s_mem, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        s_mem_all = pd.DataFrame(df_s_mem.values[:, 1])
        # print(itn_time_all)

        s_mem_all = s_mem_all.to_numpy()

        s_mem = s_mem_all[len(s_mem_all) - 5 : len(s_mem_all)]

        return s_mem

    def get_server_gpu():
        file_read_s_gpu = "server_gpu.csv"
        df_s_gpu = pd.read_csv(file_read_s_gpu, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        s_gpu_all = pd.DataFrame(df_s_gpu.values[:, 1])
        # print(itn_time_all)

        s_gpu_all = s_gpu_all.to_numpy()

        s_gpu = s_gpu_all[len(s_gpu_all) - 5 : len(s_gpu_all)]

        return s_gpu

    def get_server_gpu_mem():
        file_read_s_gpu_mem = "server_gpu_mem.csv"
        df_s_gpu_mem = pd.read_csv(file_read_s_gpu_mem, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        s_gpu_mem_all = pd.DataFrame(df_s_gpu_mem.values[:, 1])
        # print(itn_time_all)

        s_gpu_mem_all = s_gpu_mem_all.to_numpy()

        s_gpu_mem = s_gpu_mem_all[len(s_gpu_mem_all) - 5 : len(s_gpu_mem_all)]

        return s_gpu_mem

    def get_server_band_in():
        file_read_s_band_in = "server_band_in.csv"
        df_s_band_in = pd.read_csv(file_read_s_band_in, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        s_band_in_all = pd.DataFrame(df_s_band_in.values[:, 1])
        # print(itn_time_all)

        s_band_in_all = s_band_in_all.to_numpy()

        s_band_in = s_band_in_all[len(s_band_in_all) - 5 : len(s_band_in_all)]

        return s_band_in

    def get_server_band_out():
        file_read_s_band_out = "server_band_out.csv"
        df_s_band_out = pd.read_csv(file_read_s_band_out, header=None)
        # print(df_itn)

        # task_id_all = pd.DataFrame(df_itn.values[:,0])
        # print(itn_time_all)

        s_band_out_all = pd.DataFrame(df_s_band_out.values[:, 1])
        # print(itn_time_all)

        s_band_out_all = s_band_out_all.to_numpy()

        s_band_out = s_band_out_all[len(s_band_out_all) - 5 : len(s_band_out_all)]

        return s_band_out

    def get_state(self, job_id):
        # interval = 60
        logger = Logger(job_name=f"job{job_id}_ssgd", file_dir=f"{file_path_prefix}/logs/job{job_id}_node_info.log").logger

        model_name = ssgd_list[self.model_list[job_id]]["model_name"]
        logger.info(f"model_name: {model_name}")

        job_placement = "job_placement:"
        # for loc in self.job_location[job_id]:
        #     job_placement += f" {loc}"
        # logger.info(job_placement)

        # socks = []
        # for e in range(self.num_workers + 1):
        #     connm = znet.SocketMsger.tcp_connect(self.slave_address[e], 9093)
        #     socks.append(connm)

        resource_info = [
            "0.0 0.0 0.0 0.0 0.0 0.0|0.0 0.0 0.0 0.0 0.0 0.0|0.0 0.0 0.0 0.0 0.0 0.0|0.0 0.0 0.0 0.0 0.0 0.0|0.0 0.0 0.0 0.0 0.0 0.0"
        ]
        agent_connm = znet.SocketMsger.tcp_connect(self.slave_address[0], 46996)
        threading.Thread(target=self.switch, args=(job_id, agent_connm, logger)).start()

        time_diff = 0.0
        # while True:
        #     sleep_time = 1.0 - time_diff
        #     if sleep_time >= 0.0:
        #         time.sleep(sleep_time)
        #     t0 = time.time()
        #     info_list = ""
        #     for e in range(self.num_workers + 1):
        #         connm: znet.SocketMsger = socks[e]
        #         # connm.send(f"job{job_id}-{self.entities[e]}")
        #         connm.send("G")
        #         info = connm.recv()
        #         info_list = f"{info_list}|{info}"
        #     info_list = info_list.strip("|")
        #     resource_info[0] = info_list
        # for e in range(self.num_workers + 1):
        #     conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     # establish connections with slaves
        #     host = self.slave_address[self.job_location[job_id][e]]
        #     conn.connect((host, 9093))

        #     # send entity info
        #     entity = f"job{job_id}-{self.entities[e]}\0"
        #     conn.send(entity.encode("utf-8"))

        #     # receive info from slaves
        #     info = conn.recv(102400).decode("utf-8")
        #     info_list = f"{info_list} {info}"

        #     conn.close()

        # if not self.complete_flag[job_id]:
        #     logger.info(f"{info_list}")
        #     # pass

        # else:
        #     logger.info(f"{info_list}")
        #     agent_connm.close()
        #     for connm in socks:
        #         connm.close()
        #     break

        # time_diff = time.time() - t0

    def delete_jobs(self):
        file_read_status = f"{file_path_prefix}/complete_status.csv"

        df = pd.read_csv(file_read_status, header=None)
        # print(df_itn)
        if len(df.index) > 0:
            index_job_name = pd.DataFrame(df.values[:, 0])
            # print(itn_time_all)

            index_worker = pd.DataFrame(df.values[:, 1])

            index_status = pd.DataFrame(df.values[:, 2])
            # print(itn_time_all)

            index_job_name = index_job_name.to_numpy()

            # index_worker =index_worker.to_numpy()

            index_status = index_status.to_numpy()

            for job_name in index_job_name:
                if job_name in self.completed_jobs:
                    skip = 1
                else:
                    self.completed_jobs.append(job_name)
                    # id_j = int(job_name[3:len(job_name)])
                    if job_name in self.current_jobs:
                        self.current_jobs.remove(job_name)

    # def delete_jobs(self):
    #     completed_jobs = []
    #     for i, j in enumerate(self.current_jobs):
    #         # check the status of job j's ps. add it to completed jobs if its status is succeeded.
    #         # job_name = f"job{j}-ps"
    #         # api_response = self.batch_v1.read_namespaced_job_status(namespace="default", name=job_name)
    #         # status = api_response.status.succeeded
    #         status = get_status()
    #         if status is not None:
    #             completed_jobs.append(j)
    #             self.completed_jobs.append(j)

    #     for j in completed_jobs:
    #         self.current_jobs.remove(j)  # remove job j
    #         self.complete_flag[j] = True  # set its complete flag to True

    #         # check status of each entity in job j and terminate the job once each entity is succeeded
    #         while True:
    #             status = 0
    #             job_name_list = []
    #             job_name_list.append(f"job{j}-ps")
    #             job_name_list.append(f"job{j}-tester")
    #             for w in range(1, self.num_workers + 1):
    #                 job_name_list.append(f"job{j}-worker{w}")

    #         #     for job in job_name_list:
    #         #         api_response = self.batch_v1.read_namespaced_job_status(namespace="default", name=job)

    #         #         if api_response.status.succeeded is not None:
    #         #             status += 1

    #         #     if status == self.num_workers + 2:
    #         #         break

    #         # os.system(f"kubectl delete -f yaml/job{j}.yaml")

    def add_jobs(self):
        t1 = time.time()
        if self.time_list:
            if t1 - self.t0 > self.time_list[0]:
                self.waiting_jobs.append(self.incoming_job)
                if self.model_list[self.incoming_job] < 8:
                    # generate yaml file for a job on CIFAR dataset
                    print("creating ")
                    job_name = f"job{self.incoming_job}"
                    model_name = ssgd_list[self.model_list[self.incoming_job]]["model_name"]
                    num_workers = self.num_workers
                    batch_size = ssgd_list[self.model_list[self.incoming_job]]["batch_size"]
                    lr = ssgd_list[self.model_list[self.incoming_job]]["lr"]

                    # generate_cifar(
                    #     job_name=f"job{self.incoming_job}",
                    #     model_name=ssgd_list[self.model_list[self.incoming_job]]["model_name"],
                    #     num_workers=self.num_workers,
                    #     # ss=ssgd_list[self.model_list[self.incoming_job]]["ss"],
                    #     batch_size=ssgd_list[self.model_list[self.incoming_job]]["batch_size"],
                    #     lr=ssgd_list[self.model_list[self.incoming_job]]["lr"],
                    # )
                else:
                    # generate yaml file for a job on wikitext-2 dataset
                    job_name = f"job{self.incoming_job}"
                    model_name = ssgd_list[self.model_list[self.incoming_job]]["model_name"]
                    num_workers = self.num_workers
                    # ss=ssgd_list[self.model_list[self.incoming_job]]["ss"],
                    batch_size = ssgd_list[self.model_list[self.incoming_job]]["batch_size"]
                    lr = ssgd_list[self.model_list[self.incoming_job]]["lr"]
                    # generate_wiki(
                    #     job_name=f"job{self.incoming_job}",
                    #     model_name=ssgd_list[self.model_list[self.incoming_job]]["model_name"],
                    #     num_workers=self.num_workers,
                    #     # ss=ssgd_list[self.model_list[self.incoming_job]]["ss"],
                    #     batch_size=ssgd_list[self.model_list[self.incoming_job]]["batch_size"],
                    #     lr=ssgd_list[self.model_list[self.incoming_job]]["lr"],
                    # )

                self.time_list.remove(self.time_list[0])  # remove the job from time_list
                self.incoming_job += 1  # job_id + 1

        for _ in range(self.max_jobs - len(self.current_jobs)):
            if self.waiting_jobs:

                new_job = self.waiting_jobs[0]  # get the first job in the waiting list
                self.current_jobs.append(new_job)  # add it on the current job list
                # os.system(f"kubectl apply -f yaml/job{new_job}.yaml")  # execute yaml file

                # command_ps = python cifar.py","--model={{model_name}}","--job_name={{job_name}}","--rank={{rank}}","--batch_size={{batch_size}}","--num_workers={{num_workers}}","--master_addr={{job_name}}","--num_epochs=100","--lr={{lr}}"
                # client_record[0].sendall(str.encode(command_ps))

                nepochs = np.random.randint(low=70, high=100)

                global gpu_dev_list_index
                gpu_id = gpu_dev_list[gpu_dev_list_index]
                gpu_dev_list_index = (gpu_dev_list_index + 1) % len(gpu_dev_list)

                command_ps = (
                    f"python {file_path_prefix}/cifar_switch.py"
                    + " --model="
                    + str(model_name)
                    + " --job_name="
                    + str(job_name)
                    + " --rank="
                    + str(0)
                    + " --batch_size="
                    + str(batch_size)
                    + " --num_workers="
                    + str(num_workers)
                    + " --master_addr="
                    + str(list_server[0])
                    + " --master_port="
                    + str(self.global_port)
                    + " --num_epochs="
                    + str(nepochs)
                    + " --lr="
                    + str(lr)
                    + " --gpu_id="
                    + str(gpu_id)
                )
                self.client_record[0].sendall(str.encode(command_ps))

                time.sleep(2)

                for rank in range(1, num_workers + 1):
                    command_worker = (
                        f"python {file_path_prefix}/cifar_switch.py"
                        + " --model="
                        + str(model_name)
                        + " --job_name="
                        + str(job_name)
                        + " --rank="
                        + str(rank)
                        + " --batch_size="
                        + str(batch_size)
                        + " --num_workers="
                        + str(num_workers)
                        + " --master_addr="
                        + str(list_server[0])
                        + " --master_port="
                        + str(self.global_port)
                        + " --num_epochs="
                        + str(nepochs)
                        + " --lr="
                        + str(lr)
                        + " --gpu_id="
                        + str(gpu_id)
                    )
                    self.client_record[self.count_gpu].sendall(str.encode(command_worker))
                    time.sleep(2)
                    self.count_gpu = (self.count_gpu + 1) % 5

                command_tester = (
                    f"python {file_path_prefix}/cifar_switch.py"
                    + " --model="
                    + str(model_name)
                    + " --job_name="
                    + str(job_name)
                    + " --rank="
                    + str(num_workers + 1)
                    + " --batch_size="
                    + str(batch_size)
                    + " --num_workers="
                    + str(num_workers)
                    + " --master_addr="
                    + str(list_server[0])
                    + " --master_port="
                    + str(self.global_port)
                    + " --num_epochs="
                    + str(nepochs)
                    + " --lr="
                    + str(lr)
                    + " --gpu_id="
                    + str(gpu_id)
                )
                self.client_record[5].sendall(str.encode(command_tester))
                time.sleep(2)

                self.global_port = self.global_port + 1

                # time.sleep(5)

                # command_worker1 = 'python all_job/'+'cifar10_'+file_name+'_w0.py --job_name=worker --task_id=0 --ps_hosts='+ps_server+':'+str(port+1)+' --worker_hosts='+w1_server+':'+str(port+2)+","+w2_server+':'+str(port+3)+' --max_steps='+str(epoch)+" --job_number="+str(itera)+" --deadline="+str(int(deadline))
                # client_record[(container_id[1])].sendall(str.encode(command_worker1))

                # time.sleep(5)

                # command_worker2 = 'python all_job/'+'cifar10_'+file_name+'_w0.py --job_name=worker --task_id=1 --ps_hosts='+ps_server+':'+str(port+1)+' --worker_hosts='+w1_server+':'+str(port+2)+","+w2_server+':'+str(port+3)+' --max_steps='+str(epoch)+" --job_number="+str(itera)+" --deadline="+str(int(deadline))
                # client_record[(container_id[2])].sendall(str.encode(command_worker2))

                self.waiting_jobs.remove(new_job)  # remove it from the current waiting list

                # get assignment of PS and workers
                # for e in range(self.num_workers + 1):
                #     self.job_location[new_job][e] = self.get_location(entity=f"job{new_job}-{self.entities[e]}")

                # get state of the new job
                self.job_thread[new_job] = threading.Thread(target=self.get_state, args=(new_job,))
                self.job_thread[new_job].start()

    def update_waiting_jobs(self):
        t1 = time.time()
        if self.time_list:
            if t1 - self.t0 > self.time_list[0]:
                self.waiting_jobs.append(self.incoming_job)
                if self.model_list[self.incoming_job] < 8:
                    # generate yaml file for a job on CIFAR dataset
                    generate_cifar(
                        job_name=f"job{self.incoming_job}",
                        model_name=ssgd_list[self.model_list[self.incoming_job]]["model_name"],
                        num_workers=self.num_workers,
                        # ss=ssgd_list[self.model_list[self.incoming_job]]["ss"],
                        batch_size=ssgd_list[self.model_list[self.incoming_job]]["batch_size"],
                        lr=ssgd_list[self.model_list[self.incoming_job]]["lr"],
                    )
                else:
                    # generate yaml file for a job on wikitext-2 dataset
                    generate_wiki(
                        job_name=f"job{self.incoming_job}",
                        model_name=ssgd_list[self.model_list[self.incoming_job]]["model_name"],
                        num_workers=self.num_workers,
                        # ss=ssgd_list[self.model_list[self.incoming_job]]["ss"],
                        batch_size=ssgd_list[self.model_list[self.incoming_job]]["batch_size"],
                        lr=ssgd_list[self.model_list[self.incoming_job]]["lr"],
                    )

                self.time_list.remove(self.time_list[0])  # remove the job from time_list
                self.incoming_job += 1  # job_id + 1

    def run(self):
        # self.update_waiting_jobs()
        # self.delete_jobs()
        self.add_jobs()
        self.delete_jobs()

        logging.info(f"current jobs: {self.current_jobs} | waiting jobs: {self.waiting_jobs}")


def main():
    ThreadCount = 0

    try:
        ServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ServerSocket.bind((host, port))
    except socket.error as e:
        print(str(e))

    print("Waitiing for a Connection..")
    ServerSocket.listen(5)

    number_of_client = 0
    client_record = []

    while True:
        Client, address = ServerSocket.accept()
        print("Connected to: " + address[0] + ":" + str(address[1]))
        start_new_thread(threaded_client, (Client,))
        ThreadCount += 1
        print("Thread Number: " + str(ThreadCount))
        client_record.append(Client)

        number_of_client = number_of_client + 1

        if number_of_client == 6:
            break

    slave_address = list_server

    logging.info("Read arriving time of jobs")
    file = open(f"{file_path_prefix}/time_list.txt", mode="r")
    t0 = time.time()

    time_list = []
    for t in file.readlines():
        hour, minute, second = t.split(":")
        start_time = 3600 * int(hour) + 60 * int(minute) + int(second)
        time_list.append(start_time)

    logging.info("Read job list")
    num_jobs = len(time_list)  # number of jobs
    num_model_types = len(ssgd_list)  # number of job types (default: 10)
    job_list = []
    model_list = []
    for j in range(num_jobs):
        # selected_model = np.random.randint(low=0, high=num_model_types)  # randomly select a job
        selected_model = j  # randomly select a job
        job_list.append(j)  # job_list: List = [0, 1, 2, ..., num_jobs]
        model_list.append(selected_model)  # add the selected job to model_list

    scheduler = Scheduler(
        max_jobs=3,
        time_list=time_list,
        job_list=job_list,
        model_list=model_list,
        slave_address=slave_address,
        client_record=client_record,
    )

    logging.info("Start!")
    while True:
        scheduler.run()
        time.sleep(2)

    # task_assign(client_record)


if __name__ == "__main__":
    main()
