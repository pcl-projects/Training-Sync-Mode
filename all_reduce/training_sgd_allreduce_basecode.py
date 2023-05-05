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

rand_seed = 218276150
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, job_name, file_path, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(job_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_path, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    @property
    def logger(self):
        return self.__logger


class Worker:
    def __init__(
        self,
        wrk_id,
        job_name,
        model_name,
        model,
        wrk_num,
        epoch_num,
        learning_rate,
        gpu_id,
        data_loader,
        wrk_pkey_lists,
        wrk_pkey_locs,
    ) -> None:
        self.wrk_id = wrk_id
        self.job_name = job_name
        self.model_name = model_name
        self.model = model
        self.model_lock = threading.Lock()
        self.wrk_num = wrk_num
        self.epoch_num = epoch_num
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.wrk_pkey_lists = wrk_pkey_lists
        self.wrk_pkey_locs = wrk_pkey_locs
        self.grad_lists = None
        self.wrk_rrefs = None
        self.pre_wrk_rref = None
        self.tester_rref = None
        self.logger = Logger(job_name=job_name, file_path=f"./training_logs/{job_name}_worker{wrk_id}.log").logger
        self.logger.info(f"Model name: {model_name}")

        self.epoch_idx = 0
        self.iter_idx = 0
        self.iter_count = 0

        # aggregate
        self.aggr_msg_q = queue.Queue()

        # param update
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def set_wrk_rrefs_and_tester_rref(self, worker_rrefs, tester_rref):
        if not isinstance(self, Worker):
            self = self.local_value()

        self.wrk_rrefs = worker_rrefs
        self.pre_wrk_rref = worker_rrefs[(self.wrk_id - 1) % self.wrk_num]
        self.tester_rref = tester_rref

    def get_model(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        model = None
        with self.model_lock:
            self.model = self.model.cpu()
            model = copy.deepcopy(self.model)
            self.model = self.model.to(self.device)

        return model

    def get_gradient_slice(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        grad_slice = self.aggr_msg_q.get()

        return grad_slice, time.time()

    def run_worker(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        self.model = self.model.to(self.device)
        for epoch_idx in range(self.epoch_num):
            self.epoch_idx = epoch_idx
            for iter_idx, (data, target) in enumerate(self.data_loader):
                self.iter_idx = iter_idx
                self.iter_count += 1

                wrk_cp_t_0 = time.time()

                # self.model = self.model.to(device)
                data, target = data.to(self.device), target.to(self.device)
                with self.model_lock:
                    # self.model = self.model.to(device)
                    self.model.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    # self.model = self.model.cpu()

                wrk_cp_t = 1000 * (time.time() - wrk_cp_t_0)

                self.grad_lists = [[None for _ in range(len(self.wrk_pkey_lists[i]))] for i in range(self.wrk_num)]
                with self.model_lock:
                    for pkey, param in self.model.named_parameters():
                        pkey_loc = self.wrk_pkey_locs[pkey]
                        self.grad_lists[pkey_loc[0]][pkey_loc[1]] = param.grad.cpu()
                # for idx in range(self.wrk_num):
                #     pkey_list = self.wrk_pkey_lists[idx]
                #     for key in pkey_list:
                #         # print(self.model.state_dict()[key].grad)
                #         self.grad_lists[idx].append(self.model.state_dict()[key].grad)

                wrk_cm_t = 0.0

                for idx in range(self.wrk_num - 1):
                    send_slice_id = (self.wrk_id - idx) % self.wrk_num
                    recv_slice_id = (send_slice_id - 1) % self.wrk_num
                    send_grad_slice = self.grad_lists[send_slice_id]
                    self.aggr_msg_q.put(send_grad_slice)
                    recv_grad_slice, wrk_cm_t_0 = self.pre_wrk_rref.rpc_sync().get_gradient_slice()
                    wrk_cm_t += 1000 * (time.time() - wrk_cm_t_0)
                    grad_list_to_add = self.grad_lists[recv_slice_id]
                    for i in range(len(grad_list_to_add)):
                        grad_list_to_add[i] += recv_grad_slice[i]
                for idx in range(self.wrk_num - 1):
                    send_slice_id = (self.wrk_id + 1 - idx) % self.wrk_num
                    recv_slice_id = (send_slice_id - 1) % self.wrk_num
                    send_grad_slice = self.grad_lists[send_slice_id]
                    self.aggr_msg_q.put(send_grad_slice)
                    recv_grad_slice, wrk_cm_t_0 = self.pre_wrk_rref.rpc_sync().get_gradient_slice()
                    wrk_cm_t += 1000 * (time.time() - wrk_cm_t_0)
                    self.grad_lists[recv_slice_id] = recv_grad_slice

                with self.model_lock:
                    for pkey, param in self.model.named_parameters():
                        pkey_loc = self.wrk_pkey_locs[pkey]
                        grad = self.grad_lists[pkey_loc[0]][pkey_loc[1]].to(self.device)
                        param.grad = grad
                    # for idx in range(self.wrk_num):
                    #     pkey_list = self.wrk_pkey_lists[idx]
                    #     grad_list = self.grad_lists[idx]
                    #     for key, grad in zip(pkey_list, grad_list):
                    #         grad = grad.to(device)
                    #         self.model.state_dict()[key].grad = grad
                    self.optimizer.step()

                rpc.rpc_async(to="manager", func=update_wrk_info, args=(self.wrk_id, wrk_cm_t, wrk_cp_t))
                stop_flag = self.tester_rref.rpc_sync().get_stop_flag()

                if stop_flag:
                    break

            if stop_flag:
                break


class Tester:
    def __init__(
        self,
        job_name,
        model_name,
        worker_num,
        gpu_id,
        worker_rrefs,
        # model_fetch_ids,
        test_batch_size,
        test_target_loss,
        test_data_loader,
        test_dataset,
        test_wrk_id,
    ) -> None:
        self.job_name = job_name
        self.model_name = model_name
        self.model = None
        self.worker_num = worker_num
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.worker_rrefs = worker_rrefs
        # self.model_fetch_ids = model_fetch_ids
        # self.cur_fetch_index = 0
        self.test_batch_size = test_batch_size
        self.test_target_loss = test_target_loss
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset
        self.test_wrk_id = test_wrk_id
        self.logger = Logger(job_name=job_name, file_path=f"./training_logs/{job_name}_tester.log").logger
        self.logger.info(f"Model name: {model_name}")

        self.stop_flag = False

    def get_stop_flag(self):
        if not isinstance(self, Tester):
            self = self.local_value()

        return self.stop_flag

    def test_model(self):
        if not isinstance(self, Tester):
            self = self.local_value()

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        time_init = time.time()
        time_0 = time_init
        # self.model = self.model.to(device)
        while True:
            time_1 = time.time()

            if time_1 - time_0 >= 40:
                time_0 = time_1

                self.model = self.worker_rrefs[self.test_wrk_id].rpc_sync().get_model().to(self.device)

                test_correct = 0.0
                test_loss = 0.0

                with torch.no_grad():
                    for _, (data, target) in enumerate(self.test_data_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        loss = criterion(output, target)

                        test_loss += loss.item()
                        _, predicted = output.max(1)
                        test_correct += predicted.eq(target).sum().item()

                    self.model = self.model.cpu()

                test_loss = test_loss * self.test_batch_size / len(self.test_data_loader.dataset)
                test_accuracy = 100.0 * test_correct / len(self.test_dataset)

                step_num = rpc.rpc_sync(to="manager", func=get_step_num)

                self.logger.info(
                    "Steps: {} | Loss: {:.4f} | Acc.: {:.4f} % | Time: {:.4f} s".format(
                        step_num, test_loss, test_accuracy, time_1 - time_init
                    )
                )

                # if test_loss <= self.test_target_loss:
                #     self.ps_rref.rpc_sync().stop_ps()
                #     break


wrk_cm_times = []
wrk_cp_times = []
step_num = 0
step_num_lock = threading.Lock()


def mngr_init_global_vars(wrk_num):
    global wrk_cm_times
    global wrk_cp_times
    wrk_cm_times = [0.0 for _ in range(wrk_num)]
    wrk_cp_times = [0.0 for _ in range(wrk_num)]


def update_wrk_info(wrk_id, wrk_cm_t, wrk_cp_t, added_step_num=1):
    global step_num
    global step_num_lock
    wrk_cm_times[wrk_id] = wrk_cm_t
    wrk_cp_times[wrk_id] = wrk_cp_t
    with step_num_lock:
        step_num += added_step_num


def get_step_num():
    global step_num
    return step_num


def main(
    job_name,
    model_name,
    rpc_rank,
    wrk_num,
    training_data_dir,
    batch_size,
    test_batch_size,
    test_target_loss,
    test_wrk_id,
    learning_rate,
    epoch_num,
    data_partitioned,
    gpu_ids,
    # model_fetch_ids,
):
    logging.basicConfig(level=logging.INFO)
    world_size = wrk_num + 2
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=0, _transports=["uv"])

    if rpc_rank == 0:  # manager
        # Partition parameters to PSs
        model = None
        if model_name == "alexnet":
            model = alexnet.AlexNet()
        elif model_name == "resnet20":
            model = resnet3.resnet20()
        elif model_name == "resnet56":
            model = resnet3.resnet56()
        elif model_name == "vgg13":
            model = vgg.VGG13()
        elif model_name == "vgg16":
            model = vgg.VGG16()
        elif model_name == "densenet121":
            model = densenet.DenseNet121()
        elif model_name == "googlenet":
            model = googlenet.GoogLeNet()
        elif model_name == "mobilenet":
            model = mobilenetv2.MobileNetV2()
        pkey_numel = {}
        key_param = {}
        wrk_param_nums = [0 for _ in range(wrk_num)]
        # ps_param_lists = [[] for _ in range(ps_num)]  # needed for ps
        # param_ps_idx = {}  # needed for ps and worker
        # param_loc_idx = {}  # needed for ps and worker
        wrk_pkey_lists = [[] for _ in range(wrk_num)]
        wrk_pkey_locs = {}
        for key, param in model.named_parameters():
            pkey_numel[key] = param.numel()
            key_param[key] = param
        pkey_numel = sorted(pkey_numel.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(pkey_numel)):
            key = pkey_numel[i][0]
            idx = np.argmin(wrk_param_nums)
            # param_ps_idx[key] = idx
            # param_loc_idx[key] = len(ps_param_lists[idx])
            wrk_param_nums[idx] += pkey_numel[i][1]
            wrk_pkey_locs[key] = (idx, len(wrk_pkey_lists[idx]))
            wrk_pkey_lists[idx].append(key)
        # Parameter partitioning done

        # Initializing all data_loaders for all workers.
        data_loaders = []
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        training_dataset = datasets.CIFAR10(root=training_data_dir, train=True, download=True, transform=transform)
        if data_partitioned == 1:
            dataset_len = len(training_dataset)
            worker_dataset_len = int((dataset_len + wrk_num) / wrk_num)
            len_list = [worker_dataset_len for _ in range(wrk_num - 1)]
            len_list.append(dataset_len - (wrk_num - 1) * worker_dataset_len)
            training_datasets = random_split(training_dataset, len_list)
            for id in range(wrk_num):
                data_loader = DataLoader(training_datasets[id], batch_size=batch_size, shuffle=True)
                data_loaders.append(data_loader)
        else:
            for _ in range(wrk_num):
                data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
                data_loaders.append(data_loader)
        # Initialized all data_loaders for all workers.

        # Initializing test_data_loader.
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
        test_dataset = datasets.CIFAR10(root=training_data_dir, train=False, download=True, transform=test_transform)
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        # Initialized test_data_loader.

        logging.info(f"{job_name} manager initializing.")
        rpc.init_rpc("manager", rank=0, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} manager initialized.")

        wrk_rrefs = []
        tester_rref = None

        for id in range(wrk_num):
            wrk_rref = rpc.remote(
                to=f"worker{id}",
                func=Worker,
                args=(
                    id,
                    job_name,
                    model_name,
                    model,
                    wrk_num,
                    epoch_num,
                    learning_rate,
                    gpu_ids[id],
                    data_loaders[id],
                    wrk_pkey_lists,
                    wrk_pkey_locs,
                ),
            )
            wrk_rrefs.append(wrk_rref)
        tester_rref = rpc.remote(
            to="tester",
            func=Tester,
            args=(
                job_name,
                model_name,
                wrk_num,
                gpu_ids[-1],
                wrk_rrefs,
                # model_fetch_ids,
                test_batch_size,
                test_target_loss,
                test_data_loader,
                test_dataset,
                test_wrk_id,
            ),
        )
        for wrk_rref in wrk_rrefs:
            wrk_rref.rpc_sync().set_wrk_rrefs_and_tester_rref(wrk_rrefs, tester_rref)

        mngr_init_global_vars(wrk_num)

        futures = []
        for id in range(wrk_num):
            futures.append(rpc.rpc_async(to=f"worker{id}", func=Worker.run_worker, args=(wrk_rrefs[id],)))
        futures.append(rpc.rpc_async(to="tester", func=Tester.test_model, args=(tester_rref,)))
        torch.futures.wait_all(futures)

        logging.info("All workers and tester complete.")
    elif rpc_rank <= wrk_num:  # workers
        logging.info(f"{job_name} worker{rpc_rank - 1} initializing.")
        rpc.init_rpc(f"worker{rpc_rank - 1}", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} worker{rpc_rank - 1} initialized.")
    else:  # tester
        logging.info(f"{job_name} tester initializing.")
        rpc.init_rpc("tester", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} tester initialized.")

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--job_name", type=str, default="job0")
    parser.add_argument("--model_name", type=str, default="alexnet")
    parser.add_argument("--rpc_rank", type=int, default=0)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--training_data_dir", type=str, default="./training_data/")
    parser.add_argument("--rpc_master_addr", type=str, default="localhost")
    parser.add_argument("--rpc_master_port", type=str, default="29600")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--test_target_loss", type=float, default=0.8)
    parser.add_argument("--test_wrk_id", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--data_partitioned", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, required=True)
    # parser.add_argument("--model_fetch_ids", type=str, default="0")

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.rpc_master_addr
    os.environ["MASTER_PORT"] = args.rpc_master_port

    main(
        args.job_name,
        args.model_name,
        args.rpc_rank,
        args.worker_num,
        args.training_data_dir,
        args.batch_size,
        args.test_batch_size,
        args.test_target_loss,
        args.test_wrk_id,
        args.learning_rate,
        args.epoch_num,
        args.data_partitioned,
        args.gpu_ids.split(","),
        # args.model_fetch_ids.split(","),
    )
