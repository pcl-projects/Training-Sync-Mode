#! /usr/bin/env python3


import argparse
import copy
import logging
import os
import random
import threading
import time

import numpy as np
import psutil
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
from zeyu_utils import net as znet

SRV_NIC = "ens3"

from models import (
    alexnet,
    densenet,
    googlenet,
    lstm,
    mobilenetv2,
    resnet3,
    transformer,
    vgg,
)

rand_seed = 218276150
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def data_process(raw_text_iter, vocab, tokenizer):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


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


class ParameterServer:
    def __init__(
        self,
        ps_id,
        job_name,
        model_name,
        ps_num,
        worker_num,
        batch_size,
        learning_rate,
        param_ps_idx,
        param_loc_idx,
        ps_params,
    ) -> None:
        self.ps_id = ps_id
        self.job_name = job_name
        self.model_name = model_name
        self.ps_num = ps_num
        self.worker_num = worker_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.param_ps_idx = param_ps_idx
        self.param_loc_idx = param_loc_idx
        self.ps_params = ps_params
        self.ps_rrefs = None
        self.worker_rrefs = None
        self.tester_rref = None
        self.logger = Logger(job_name=job_name, file_path=f"./training_logs/{job_name}_ps{ps_id}.log").logger
        self.logger.info(f"Model name: {model_name}")

        self.ps_comp_future = torch.futures.Future()
        self.sync_mode = 0
        self.sync_lock = threading.Lock()
        self.sync_worker_count = 0

        for param in self.ps_params:
            param.grad = None
        # self.optimizer = optim.SGD(self.ps_params, lr=learning_rate, momentum=0.9)
        self.optimizer = optim.SGD(self.ps_params, lr=learning_rate)

        self.step_num = 0

        # For worker/PS computation time and communication time
        self.ps_cp_t = [0.0 for _ in range(worker_num)]
        self.wrk_cp_t = [0.0 for _ in range(worker_num)]
        self.in_cm_t = [0.0 for _ in range(worker_num)]
        self.out_cm_t = [0.0 for _ in range(worker_num)]

        self.epoch_iter_idx_count = [(0, 0, 0) for _ in range(worker_num)]

    def set_ps_worker_rrefs_and_tester_rref(self, ps_rrefs, worker_rrefs, tester_rref):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        self.ps_rrefs = ps_rrefs
        self.worker_rrefs = worker_rrefs
        self.tester_rref = tester_rref

    def set_sync_mode(self, sync_mode):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        self.sync_mode = sync_mode

    def get_step_num(self):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        return self.step_num

    def get_ps_params(self):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        return self.ps_params

    @rpc.functions.async_execution
    def ps_computing(self, worker_id, epoch_iter_idx_count, gradients, wrk_cp_t, ps2wrk_out_cm_t, in_cm_t_0):
        in_cm_t = 1000 * (time.time() - in_cm_t_0)

        if not isinstance(self, ParameterServer):
            self = self.local_value()

        with self.sync_lock:
            sync_mode = int(self.sync_mode)

            self.wrk_cp_t[worker_id] = wrk_cp_t
            self.in_cm_t[worker_id] = in_cm_t
            self.out_cm_t[worker_id] = ps2wrk_out_cm_t
            self.epoch_iter_idx_count[worker_id] = epoch_iter_idx_count

            self.step_num += 1

            self.logger.info(f"wrk {worker_id} {in_cm_t:.3f} {ps2wrk_out_cm_t:.3f}")

            for param, grad in zip(self.ps_params, gradients):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad
            self.sync_worker_count += 1

            ft = self.ps_comp_future

            if sync_mode == 1 or self.sync_worker_count >= self.worker_num:
                ps_cp_t_0 = time.time()

                if sync_mode == 0 or (sync_mode == 1 and self.sync_worker_count > 1):
                    for param in self.ps_params:
                        param.grad = param.grad / self.sync_worker_count

                self.optimizer.step()
                # self.optimizer.zero_grad()
                for param in self.ps_params:
                    param.grad = None
                self.sync_worker_count = 0

                ps_cp_t = 1000 * (time.time() - ps_cp_t_0)

                if sync_mode == 0:
                    for i in range(self.worker_num):
                        self.ps_cp_t[i] = ps_cp_t
                else:
                    self.ps_cp_t[worker_id] = ps_cp_t

                ft.set_result((self.ps_params, time.time()))
                self.ps_comp_future = torch.futures.Future()

        return ft


class Worker:
    def __init__(
        self,
        worker_id,
        job_name,
        model_name,
        model,
        ps_num,
        worker_num,
        training_data_dir,
        batch_size,
        epoch_num,
        gpu_id,
        data_loader,
        grad_lists,
        param_lists,
        param_ps_idx,
        param_loc_idx,
    ) -> None:
        self.worker_id = worker_id
        self.job_name = job_name
        self.model_name = model_name
        self.model = model
        self.ps_num = ps_num
        self.worker_num = worker_num
        self.training_data_dir = training_data_dir
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.gpu_id = gpu_id
        self.data_loader = data_loader
        self.grad_lists = grad_lists
        self.param_lists = param_lists
        self.param_ps_idx = param_ps_idx
        self.param_loc_idx = param_loc_idx
        self.ps_rrefs = None
        self.worker_rrefs = None
        self.tester_rref = None
        self.logger = Logger(job_name=job_name, file_path=f"./training_logs/{job_name}_worker{worker_id}.log").logger
        self.logger.info(f"Model name: {model_name}")

        self.epoch_idx = 0
        self.iter_idx = 0
        self.iter_count = 0

        self.out_cm_t = [0.0 for _ in range(ps_num)]

    def set_ps_worker_rrefs_and_tester_rref(self, ps_rrefs, worker_rrefs, tester_rref):
        if not isinstance(self, Worker):
            self = self.local_value()

        self.ps_rrefs = ps_rrefs
        self.worker_rrefs = worker_rrefs
        self.tester_rref = tester_rref

    def run_ps_computing(self, ps_id, wrk_cp_t, ps2wrk_out_cm_t):
        if not isinstance(self, Worker):
            self = self.local_value()

        ps_rref = self.ps_rrefs[ps_id]
        ps_params, out_cm_t_0 = ps_rref.rpc_sync().ps_computing(
            self.worker_id,
            (self.epoch_idx, self.iter_idx, self.iter_count),
            self.grad_lists[ps_id],
            wrk_cp_t,
            ps2wrk_out_cm_t,
            time.time(),
        )
        out_cm_t = 1000 * (time.time() - out_cm_t_0)
        self.out_cm_t[ps_id] = out_cm_t
        self.param_lists[ps_id] = ps_params

    def run_worker(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        train_iter = WikiText2(root=self.training_data_dir, split="train")
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        ntokens = len(vocab)
        bptt = 35

        train_iter, val_iter, test_iter = WikiText2(root=self.training_data_dir)
        train_data = data_process(train_iter, vocab, tokenizer)
        train_data = batchify(train_data, self.batch_size)

        self.model = self.model.to(device)
        time0 = time.time()
        cpu0 = psutil.cpu_percent()
        bwin0 = psutil.net_io_counters(pernic=True)[SRV_NIC].bytes_recv
        bwout0 = psutil.net_io_counters(pernic=True)[SRV_NIC].bytes_sent
        for epoch_idx in range(self.epoch_num):
            self.epoch_idx = epoch_idx

            if self.model_name == "lstm":
                hidden = self.model.init_hidden(self.batch_size)

            if self.model_name == "lstm" or self.model_name == "transformer":
                enumerator = enumerate(range(0, train_data.size(0) - 1, bptt))
            else:
                enumerator = enumerate(self.data_loader)

            for iter_idx, x_pack in enumerator:
                self.iter_idx = iter_idx
                self.iter_count += 1

                if self.model_name == "lstm" or self.model_name == "transformer":
                    i = x_pack
                    data, target = get_batch(train_data, i, bptt)
                else:
                    data, target = x_pack

                wrk_cp_t_0 = time.time()

                # self.model = self.model.to(device)
                data, target = data.to(device), target.to(device)
                self.model.zero_grad()
                if self.model_name == "lstm":
                    hidden = repackage_hidden(hidden)
                    output, hidden = self.model(data, hidden)
                elif self.model_name == "transformer":
                    output = self.model(data)
                    output = output.view(-1, ntokens)
                else:
                    output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                # self.model = self.model.cpu()

                for key, param in self.model.named_parameters():
                    self.grad_lists[self.param_ps_idx[key]][self.param_loc_idx[key]] = param.grad.cpu()

                wrk_cp_t = 1000 * (time.time() - wrk_cp_t_0)

                self.logger.info(f"comp. {wrk_cp_t:.3f}")

                threads = []
                out_cm_t_copy = copy.deepcopy(self.out_cm_t)
                for ps_id in range(self.ps_num):
                    thrd = threading.Thread(target=self.run_ps_computing, args=(ps_id, wrk_cp_t, out_cm_t_copy[ps_id]))
                    thrd.start()
                    threads.append(thrd)
                for thrd in threads:
                    thrd.join()

                for key in self.param_ps_idx:
                    param = self.param_lists[self.param_ps_idx[key]][self.param_loc_idx[key]]  # .to(device)
                    self.model.state_dict()[key].copy_(param)

                stop_flag = self.tester_rref.rpc_sync().get_stop_flag()

                time1 = time.time()
                bwin1 = psutil.net_io_counters(pernic=True)[SRV_NIC].bytes_recv
                bwout1 = psutil.net_io_counters(pernic=True)[SRV_NIC].bytes_sent
                cpu1 = psutil.cpu_percent()
                time_diff = time1 - time0
                bwin = (bwin1 - bwin0) / time_diff / 1024 / 1024
                bwout = (bwout1 - bwout0) / time_diff / 1024 / 1024
                time0 = time1
                bwin0 = bwin1
                bwout0 = bwout1

                self.logger.info(f"res. {cpu1} {bwin:.3f} {bwout:.3f}")

                if stop_flag:
                    break

            if stop_flag:
                break


class Tester:
    def __init__(
        self,
        job_name,
        model_name,
        model,
        ps_num,
        worker_num,
        testing_data_dir,
        gpu_id,
        ps_rrefs,
        worker_rrefs,
        # model_fetch_ids,
        test_batch_size,
        test_target_loss,
        test_data_loader,
        test_dataset,
        param_ps_idx,
        param_loc_idx,
    ) -> None:
        self.job_name = job_name
        self.model_name = model_name
        self.model = model
        self.ps_num = ps_num
        self.worker_num = worker_num
        self.testing_data_dir = testing_data_dir
        self.gpu_id = gpu_id
        self.ps_rrefs = ps_rrefs
        self.worker_rrefs = worker_rrefs
        # self.model_fetch_ids = model_fetch_ids
        # self.cur_fetch_index = 0
        self.test_batch_size = test_batch_size
        self.test_target_loss = test_target_loss
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset
        self.param_ps_idx = param_ps_idx
        self.param_loc_idx = param_loc_idx
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

        # device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        # criterion = nn.CrossEntropyLoss()
        # criterion = criterion.to(device)

        # train_iter = WikiText2(root=self.testing_data_dir, split="train")
        # tokenizer = get_tokenizer("basic_english")
        # vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
        # vocab.set_default_index(vocab["<unk>"])
        # ntokens = len(vocab)
        # bptt = 35

        # train_iter, val_iter, test_iter = WikiText2(root=self.testing_data_dir)
        # val_data = data_process(val_iter, vocab, tokenizer)
        # val_data = batchify(val_data, self.test_batch_size)

        # time_init = time.time()
        # time_0 = time_init
        # self.model = self.model.to(device)
        # while True:
        #     time_1 = time.time()

        #     if time_1 - time_0 >= 40:
        #         time_0 = time_1

        #         # fetch_index = self.cur_fetch_index
        #         # if fetch_index == len(self.model_fetch_ids) - 1:
        #         #     self.cur_fetch_index = 0
        #         # else:
        #         #     self.cur_fetch_index += 1
        #         futures = []
        #         for i in range(self.ps_num):
        #             ps_rref = self.ps_rrefs[i]
        #             futures.append(ps_rref.rpc_async().get_ps_params())
        #         ps_param_lists = []
        #         for future in futures:
        #             ps_params = future.wait()
        #             ps_param_lists.append(ps_params)
        #         # self.model = self.model.cpu()
        #         for key in self.param_ps_idx:
        #             param = ps_param_lists[self.param_ps_idx[key]][self.param_loc_idx[key]]  # .to(device)
        #             self.model.state_dict()[key].copy_(param)
        #         # self.model = self.model.to(device)

        #         test_correct = 0.0
        #         test_loss = 0.0

        #         if self.model_name == "lstm":
        #             with torch.no_grad():
        #                 hidden = self.model.init_hidden(self.test_batch_size)
        #                 for _, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
        #                     data, targets = get_batch(val_data, i, bptt)
        #                     data, targets = data.to(device), targets.to(device)
        #                     hidden = repackage_hidden(hidden)
        #                     output, hidden = self.model(data, hidden)
        #                     loss = criterion(output, targets)
        #                     test_loss += len(data) * loss.item()
        #         elif self.model_name == "transformer":
        #             with torch.no_grad():
        #                 for _, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
        #                     data, targets = get_batch(val_data, i, bptt)
        #                     data, targets = data.to(device), targets.to(device)
        #                     output = self.model(data)
        #                     output = output.view(-1, ntokens)
        #                     loss = criterion(output, targets)
        #                     test_loss += len(data) * loss.item()
        #         else:
        #             with torch.no_grad():
        #                 for _, (data, target) in enumerate(self.test_data_loader):
        #                     data, target = data.to(device), target.to(device)
        #                     output = self.model(data)
        #                     loss = criterion(output, target)
        #                     test_loss += loss.item()
        #                     _, predicted = output.max(1)
        #                     test_correct += predicted.eq(target).sum().item()

        #         if self.model_name == "lstm" or self.model_name == "transformer":
        #             test_loss /= len(val_data) - 1
        #         else:
        #             test_loss = test_loss * self.test_batch_size / len(self.test_data_loader.dataset)
        #             test_accuracy = 100.0 * test_correct / len(self.test_dataset)

        #         step_num = self.ps_rrefs[0].rpc_sync().get_step_num()

        #         if self.model_name == "lstm" or self.model_name == "transformer":
        #             self.logger.info(
        #                 "Steps: {} | Loss: {:.4f} | Time: {:.4f} s".format(step_num, test_loss, time_1 - time_init)
        #             )
        #         else:
        #             self.logger.info(
        #                 "Steps: {} | Loss: {:.4f} | Acc.: {:.4f} % | Time: {:.4f} s".format(
        #                     step_num, test_loss, test_accuracy, time_1 - time_init
        #                 )
        #             )

        #         # if test_loss <= self.test_target_loss:
        #         #     self.ps_rref.rpc_sync().stop_ps()
        #         #     break


def main(
    job_name,
    model_name,
    rpc_rank,
    ps_num,
    worker_num,
    training_data_dir,
    batch_size,
    test_batch_size,
    test_target_loss,
    learning_rate,
    epoch_num,
    data_partitioned,
    gpu_ids,
    # model_fetch_ids,
):
    logging.basicConfig(level=logging.INFO)
    world_size = ps_num + worker_num + 2
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
        elif model_name == "lstm":
            model = lstm.RNNModel(rnn_type="LSTM", ntoken=28782, ninp=200, nhid=200, nlayers=2, dropout=0)
        elif model_name == "transformer":
            model = transformer.TransformerModel(ntoken=28782, ninp=200, nhead=8, nhid=200, nlayers=2, dropout=0)
        param_numel = {}
        key_param = {}
        ps_param_nums = [0 for _ in range(ps_num)]
        ps_param_lists = [[] for _ in range(ps_num)]  # needed for ps
        param_ps_idx = {}  # needed for ps and worker
        param_loc_idx = {}  # needed for ps and worker
        for key, param in model.named_parameters():
            param_numel[key] = param.numel()
            key_param[key] = param
        param_numel = sorted(param_numel.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(param_numel)):
            key = param_numel[i][0]
            idx = np.argmin(ps_param_nums)
            param_ps_idx[key] = idx
            param_loc_idx[key] = len(ps_param_lists[idx])
            ps_param_nums[idx] += param_numel[i][1]
            ps_param_lists[idx].append(key_param[key])
        param_or_grad_lists = []
        for ps_params in ps_param_lists:
            param_or_grad_lists.append([None for _ in range(len(ps_params))])
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
            worker_dataset_len = int((dataset_len + worker_num) / worker_num)
            len_list = [worker_dataset_len for _ in range(worker_num - 1)]
            len_list.append(dataset_len - (worker_num - 1) * worker_dataset_len)
            training_datasets = random_split(training_dataset, len_list)
            for id in range(worker_num):
                data_loader = DataLoader(training_datasets[id], batch_size=batch_size, shuffle=True)
                data_loaders.append(data_loader)
        else:
            for _ in range(worker_num):
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

        ps_rrefs = []
        worker_rrefs = []
        tester_rref = None

        for id in range(ps_num):
            ps_rref = rpc.remote(
                to=f"ps{id}",
                func=ParameterServer,
                args=(
                    id,
                    job_name,
                    model_name,
                    ps_num,
                    worker_num,
                    batch_size,
                    learning_rate,
                    param_ps_idx,
                    param_loc_idx,
                    ps_param_lists[id],
                ),
            )
            ps_rrefs.append(ps_rref)
        for id in range(worker_num):
            worker_rref = rpc.remote(
                to=f"worker{id}",
                func=Worker,
                args=(
                    id,
                    job_name,
                    model_name,
                    model,
                    ps_num,
                    worker_num,
                    training_data_dir,
                    batch_size,
                    epoch_num,
                    gpu_ids[id],
                    data_loaders[id],
                    param_or_grad_lists,
                    param_or_grad_lists,
                    param_ps_idx,
                    param_loc_idx,
                ),
            )
            worker_rrefs.append(worker_rref)
        tester_rref = rpc.remote(
            to="tester",
            func=Tester,
            args=(
                job_name,
                model_name,
                model,
                ps_num,
                worker_num,
                training_data_dir,
                gpu_ids[-1],
                ps_rrefs,
                worker_rrefs,
                # model_fetch_ids,
                test_batch_size,
                test_target_loss,
                test_data_loader,
                test_dataset,
                param_ps_idx,
                param_loc_idx,
            ),
        )
        for ps_rref in ps_rrefs:
            ps_rref.rpc_sync().set_ps_worker_rrefs_and_tester_rref(ps_rrefs, worker_rrefs, tester_rref)
        for worker_rref in worker_rrefs:
            worker_rref.rpc_sync().set_ps_worker_rrefs_and_tester_rref(ps_rrefs, worker_rrefs, tester_rref)

        futures = []
        for id in range(worker_num):
            futures.append(rpc.rpc_async(to=f"worker{id}", func=Worker.run_worker, args=(worker_rrefs[id],)))
        futures.append(rpc.rpc_async(to="tester", func=Tester.test_model, args=(tester_rref,)))
        torch.futures.wait_all(futures)

        logging.info("All workers and tester complete.")
    elif rpc_rank <= ps_num:  # ps-s
        logging.info(f"{job_name} ps{rpc_rank - 1} initializing.")
        rpc.init_rpc(f"ps{rpc_rank - 1}", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} ps{rpc_rank - 1} initialized.")
    elif rpc_rank <= ps_num + worker_num:  # workers
        logging.info(f"{job_name} worker{rpc_rank - ps_num - 1} initializing.")
        rpc.init_rpc(
            f"worker{rpc_rank - ps_num - 1}", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options
        )
        logging.info(f"{job_name} worker{rpc_rank - ps_num - 1} initialized.")
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
    parser.add_argument("--ps_num", type=int, default=1)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--training_data_dir", type=str, default="./training_data/")
    parser.add_argument("--rpc_master_addr", type=str, default="localhost")
    parser.add_argument("--rpc_master_port", type=str, default="29600")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--test_target_loss", type=float, default=0.8)
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
        args.ps_num,
        args.worker_num,
        args.training_data_dir,
        args.batch_size,
        args.test_batch_size,
        args.test_target_loss,
        args.learning_rate,
        args.epoch_num,
        args.data_partitioned,
        args.gpu_ids.split(","),
        # args.model_fetch_ids.split(","),
    )
