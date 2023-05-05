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
import psutil
import regression as regr
import shallow_reg_lstm as sr_lstm
import svm
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from pgns import PGNS
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
from zeyu_utils import net as znet

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


job_launcher_addr = "172.31.92.17"


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
        server_id,
        wrk_pack_toggle,
        training_data_dir,
        batch_size,
        data_loader,
        wrk_pkey_lists,
        wrk_pkey_locs,
    ) -> None:
        self.wrk_id = wrk_id
        self.ring_wrk_id = wrk_id
        self.job_name = job_name
        self.model_name = model_name
        self.model = model
        self.model_lock = threading.Lock()
        self.wrk_num = wrk_num
        self.ring_wrk_num = wrk_num
        self.epoch_num = epoch_num
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.server_id = int(server_id)
        self.wrk_pack_toggle = int(wrk_pack_toggle)
        self.training_data_dir = training_data_dir
        self.batch_size = batch_size
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

        self.wrk_cm_t = 0.0

        # aggregate
        self.aggr_msg_q = queue.Queue()

        # param update
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.future_iter_sync = None
        self.is_ring_wrk = True

        # child
        self.parent_wrk_id = 0  # only valid when self.is_ring_wrk is False
        self.parent_param_q = queue.Queue()

        # parent
        self.child_grad_q = queue.Queue()

        self.child_wrk_reset_confirm = queue.Queue()
        self.child_wrk_use_new_policy = queue.Queue()

        # ml part
        self.ava_cpu_list = [0.0 for _ in range(100)]
        self.ava_bw_list = [0.0 for _ in range(100)]
        self.model_ava_cpu = None
        self.model_ava_bw = None
        self.model_ava_cpu_in_gpu = False
        self.model_ava_bw_in_gpu = False
        self.next_ava_cpu = None
        self.next_ava_bw = None
        # ===
        self.ava_cpu_bw_list = [[0.0, 0.0] for _ in range(100)]
        self.iter_time_list = [0.0 for _ in range(100)]
        self.model_iter_time = None
        self.model_iter_time_in_gpu = False
        self.mean_list = None
        self.std_list = None
        self.next_iter_time = None

        # overhead
        self.oh_res_pred_time = 0
        self.oh_res_pred_count = 0
        self.oh_itert_pred_time = 0
        self.oh_itert_pred_count = 0
        self.oh_heuristic_time = 0
        self.oh_heuristic_count = 0
        self.oh_ml_time = 0
        self.oh_ml_count = 0
        self.oh_hybrid_time = 0
        self.oh_hybrid_count = 0
        self.oh_ps_wrk_pack_time = 0
        self.oh_ps_wrk_pack_count = 0
        self.oh_wrk_wrk_pack_time = 0
        self.oh_wrk_wrk_pack_count = 0
        self.oh_tree_time = 0
        self.oh_tree_count = 0

    def _train_sr_lstm_model_thread(self, res_type):
        device = torch.device("cpu")
        if res_type == "cpu":
            ava_cpu_list = copy.deepcopy(self.ava_cpu_list)
            model_ava_cpu = copy.deepcopy(self.model_ava_cpu)
            if model_ava_cpu is not None:
                model_ava_cpu.set_device(device)
                model_ava_cpu = model_ava_cpu.to(device)
            model_ava_cpu_fut = rpc.rpc_async(to="manager", func=train_sr_lstm_model, args=(ava_cpu_list, model_ava_cpu))
            model_ava_cpu = model_ava_cpu_fut.wait()
            while True:
                if self.model_ava_cpu_in_gpu is False:
                    self.model_ava_cpu = model_ava_cpu
                    break
        elif res_type == "bw":
            ava_bw_list = copy.deepcopy(self.ava_bw_list)
            model_ava_bw = copy.deepcopy(self.model_ava_bw)
            if model_ava_bw is not None:
                model_ava_bw.set_device(device)
                model_ava_bw = model_ava_bw.to(device)
            model_ava_bw_fut = rpc.rpc_async(to="manager", func=train_sr_lstm_model, args=(ava_bw_list, model_ava_bw))
            model_ava_bw = model_ava_bw_fut.wait()
            while True:
                if self.model_ava_bw_in_gpu is False:
                    self.model_ava_bw = model_ava_bw
                    break

    def train_sr_lstm_model(self, res_type):
        threading.Thread(target=self._train_sr_lstm_model_thread, args=(res_type,)).start()

    def _train_reg_model_thread(self):
        model_iter_time = copy.deepcopy(self.model_iter_time)
        ava_cpu_bw_list = copy.deepcopy(self.ava_cpu_bw_list)
        iter_time_list = copy.deepcopy(self.iter_time_list)
        if model_iter_time is not None:
            model_iter_time = model_iter_time.cpu()
        future = rpc.rpc_async(to="manager", func=train_reg_model, args=(ava_cpu_bw_list, iter_time_list, model_iter_time))
        model_iter_time, self.mean_list, self.std_list = future.wait()
        while True:
            if self.model_iter_time_in_gpu is False:
                self.model_iter_time = model_iter_time
                break

    def train_reg_model(self):
        threading.Thread(target=self._train_reg_model_thread).start()

    def update_optimizer(self, lr):
        if not isinstance(self, Worker):
            self = self.local_value()
        with self.model_lock:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def let_child_wrk_use_new_policy(self):
        if not isinstance(self, Worker):
            self = self.local_value()
        self.child_wrk_use_new_policy.put(0)

    def update_wrk_sync_info(
        self, is_ring_wrk, parent_id, ring_wrk_num, ring_wrk_id, pre_wrk_rref, wrk_pkey_lists, wrk_pkey_locs
    ):
        if not isinstance(self, Worker):
            self = self.local_value()
        self.is_ring_wrk = is_ring_wrk
        self.parent_wrk_id = parent_id
        self.ring_wrk_num = ring_wrk_num
        self.ring_wrk_id = ring_wrk_id
        self.pre_wrk_rref = pre_wrk_rref
        self.wrk_pkey_lists = wrk_pkey_lists
        self.wrk_pkey_locs = wrk_pkey_locs
        qsize = self.parent_param_q.qsize()
        for _ in range(qsize):
            self.parent_param_q.get()
        qsize = self.child_grad_q.qsize()
        for _ in range(qsize):
            self.child_grad_q.get()

    def child2parent_grads(self, wrk_id, grad_lists):
        if not isinstance(self, Worker):
            self = self.local_value()
        self.child_grad_q.put((wrk_id, grad_lists))

    def parent2child_params(self, params, wrk_cm_t_0):
        if not isinstance(self, Worker):
            self = self.local_value()
        wrk_cm_t = 1000 * (time.time() - wrk_cm_t_0)
        self.parent_param_q.put((params, wrk_cm_t))

    def reset_child_wrk(self):
        if not isinstance(self, Worker):
            self = self.local_value()
        self.parent_param_q.put("R")
        self.child_wrk_reset_confirm.get()

    def set_wrk_rrefs_and_tester_rref(self, worker_rrefs, tester_rref):
        if not isinstance(self, Worker):
            self = self.local_value()

        self.wrk_rrefs = worker_rrefs
        self.pre_wrk_rref = worker_rrefs[(self.wrk_id - 1) % self.wrk_num]
        self.tester_rref = tester_rref

    def get_model(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        if self.model_name == "transformer":
            params = []
            with self.model_lock:
                for key, _ in self.model.named_parameters():
                    params.append(self.model.state_dict()[key].cpu())
            return params

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

        # worker packing sync
        if self.wrk_pack_toggle == 1:
            wrk_pack_connm = znet.SocketMsger.tcp_connect(job_launcher_addr, 62232)
            wrk_pack_connm.send(("INIT", self.server_id, self.gpu_id))

        self.model = self.model.to(self.device)

        if self.model_name == "lstm" or self.model_name == "transformer":
            train_iter = WikiText2(root=self.training_data_dir, split="train")
            tokenizer = get_tokenizer("basic_english")
            vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
            vocab.set_default_index(vocab["<unk>"])
            ntokens = len(vocab)
            bptt = 35

            train_iter, val_iter, test_iter = WikiText2(root=self.training_data_dir)
            train_data = data_process(train_iter, vocab, tokenizer)
            train_data = batchify(train_data, self.batch_size)

        psutil.cpu_percent()
        bw0 = psutil.net_io_counters()

        for epoch_idx in range(self.epoch_num):
            self.epoch_idx = epoch_idx
            hidden = None
            enumerator = None
            if self.model_name == "lstm":
                hidden = self.model.init_hidden(self.batch_size)
                enumerator = enumerate(range(0, train_data.size(0) - 1, bptt))
            elif self.model_name == "transformer":
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

                data, target = data.to(self.device), target.to(self.device)
                with self.model_lock:
                    if self.wrk_pack_toggle == 1:
                        while True:
                            wrk_pack_connm.send(("CHK",))
                            gpu_in_use = int(wrk_pack_connm.recv())
                            if gpu_in_use == 1:
                                time.sleep(0.1)
                            else:
                                wrk_pack_connm.send(("SET", 1))
                                break
                    if self.model_name == "lstm":
                        hidden = repackage_hidden(hidden)
                        self.model.zero_grad()
                        output, hidden = self.model(data, hidden)
                        loss = criterion(output, target)
                        loss.backward()
                    elif self.model_name == "transformer":
                        self.model.zero_grad()
                        output = self.model(data)
                        output = output.view(-1, ntokens)
                        loss = criterion(output, target)
                        loss.backward()
                    else:
                        self.model.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        loss.backward()
                    if self.wrk_pack_toggle == 1:
                        wrk_pack_connm.send(("SET", 0))

                wrk_cp_t = 1000 * (time.time() - wrk_cp_t_0)

                # for ring worker and child worker
                if self.is_ring_wrk:
                    # one iteration sync
                    if self.future_iter_sync is not None:
                        self.future_iter_sync.wait()
                if self.is_ring_wrk:
                    # prepare grad lists
                    self.grad_lists = [[None for _ in range(len(self.wrk_pkey_lists[i]))] for i in range(self.ring_wrk_num)]
                    with self.model_lock:
                        for pkey, param in self.model.named_parameters():
                            pkey_loc = self.wrk_pkey_locs[pkey]
                            self.grad_lists[pkey_loc[0]][pkey_loc[1]] = param.grad.cpu()
                    # add children's grads
                    wrks_to_send_params = []
                    qsize = self.child_grad_q.qsize()
                    for _ in range(qsize):
                        msg = self.child_grad_q.get()
                        wid = msg[0]
                        wrks_to_send_params.append(self.wrk_rrefs[wid])
                        glists = msg[1]
                        for idx0 in range(len(self.grad_lists)):
                            for idx1 in range(len(self.grad_lists[idx0])):
                                self.grad_lists[idx0][idx1] += glists[idx0][idx1]
                    # define wrk cm t
                    self.wrk_cm_t = 0.0
                    # exchange gradients with other workers
                    for idx in range(self.ring_wrk_num - 1):
                        send_slice_id = (self.ring_wrk_id - idx) % self.ring_wrk_num
                        recv_slice_id = (send_slice_id - 1) % self.ring_wrk_num
                        send_grad_slice = self.grad_lists[send_slice_id]
                        self.aggr_msg_q.put(send_grad_slice)
                        recv_grad_slice, wrk_cm_t_0 = self.pre_wrk_rref.rpc_sync().get_gradient_slice()
                        self.wrk_cm_t += 1000 * (time.time() - wrk_cm_t_0)
                        grad_list_to_add = self.grad_lists[recv_slice_id]
                        for i in range(len(grad_list_to_add)):
                            grad_list_to_add[i] += recv_grad_slice[i]
                    for idx in range(self.ring_wrk_num - 1):
                        send_slice_id = (self.ring_wrk_id + 1 - idx) % self.ring_wrk_num
                        recv_slice_id = (send_slice_id - 1) % self.ring_wrk_num
                        send_grad_slice = self.grad_lists[send_slice_id]
                        self.aggr_msg_q.put(send_grad_slice)
                        recv_grad_slice, wrk_cm_t_0 = self.pre_wrk_rref.rpc_sync().get_gradient_slice()
                        self.wrk_cm_t += 1000 * (time.time() - wrk_cm_t_0)
                        self.grad_lists[recv_slice_id] = recv_grad_slice
                    # update model parameter
                    with self.model_lock:
                        for pkey, param in self.model.named_parameters():
                            pkey_loc = self.wrk_pkey_locs[pkey]
                            grad = self.grad_lists[pkey_loc[0]][pkey_loc[1]].to(self.device)
                            param.grad = grad
                        self.optimizer.step()
                    # send updated params to children
                    params = []
                    for pkey, _ in self.model.named_parameters():
                        params.append(self.model.state_dict()[pkey].cpu())
                    for wref in wrks_to_send_params:
                        wref.rpc_async().parent2child_params(params, time.time())
                    # update wrk info
                    self.future_iter_sync = rpc.rpc_async(
                        to="manager",
                        func=update_wrk_info,
                        args=(self.wrk_id, self.wrk_cm_t, wrk_cp_t, self.is_ring_wrk),
                    )
                else:  # for child worker
                    # prepare grad lists
                    self.grad_lists = [[None for _ in range(len(self.wrk_pkey_lists[i]))] for i in range(self.ring_wrk_num)]
                    with self.model_lock:
                        for pkey, param in self.model.named_parameters():
                            pkey_loc = self.wrk_pkey_locs[pkey]
                            self.grad_lists[pkey_loc[0]][pkey_loc[1]] = param.grad.cpu()
                    # send gradients to parent and wait for updated parameters
                    p_wrk_ref = self.wrk_rrefs[self.parent_wrk_id]
                    p_wrk_ref.rpc_sync().child2parent_grads(self.wrk_id, self.grad_lists)
                    msg = self.parent_param_q.get()
                    if isinstance(msg, str) and msg == "R":
                        self.child_wrk_reset_confirm.put(0)
                        self.child_wrk_use_new_policy.get()
                    else:
                        params = msg[0]
                        self.wrk_cm_t = msg[1]
                        for (pkey, _), param in zip(self.model.named_parameters(), params):
                            self.model.state_dict()[pkey].copy_(param)
                    self.future_iter_sync = rpc.rpc_async(
                        to="manager",
                        func=update_wrk_info,
                        args=(self.wrk_id, self.wrk_cm_t, wrk_cp_t, False),
                    )

                # ml part
                ava_cpu = psutil.cpu_percent()
                bw1 = psutil.net_io_counters()
                ava_bw = ((bw1.bytes_recv - bw0.bytes_recv) + (bw1.bytes_sent - bw0.bytes_sent)) / 2048.0 / 1024.0
                iter_time = wrk_cp_t + self.wrk_cm_t
                self.ava_cpu_list = self.ava_cpu_list[1:]
                self.ava_cpu_list.append(ava_cpu)
                self.ava_bw_list = self.ava_bw_list[1:]
                self.ava_bw_list.append(ava_bw)
                self.ava_cpu_bw_list = self.ava_cpu_bw_list[1:]
                self.ava_cpu_bw_list.append([ava_cpu, ava_bw])
                self.iter_time_list = self.iter_time_list[1:]
                self.iter_time_list.append(iter_time)
                if self.iter_count >= 200 and (self.iter_count - 200) % 10000 == 0:
                    # if self.iter_count == 500:
                    self.train_sr_lstm_model("cpu")
                    self.train_sr_lstm_model("bw")
                if self.iter_count >= 3000 and (self.iter_count - 3000) % 10000 == 0:
                    # if self.iter_count == 3300:
                    self.train_reg_model()
                if self.model_ava_cpu is not None and self.model_ava_bw is not None:
                    time_0 = time.time()
                    self.model_ava_cpu_in_gpu = True
                    self.next_ava_cpu = sr_lstm.predict(self.model_ava_cpu, self.ava_cpu_list, self.device)
                    self.model_ava_cpu_in_gpu = False
                    self.model_ava_bw_in_gpu = True
                    self.next_ava_bw = sr_lstm.predict(self.model_ava_bw, self.ava_bw_list, self.device)
                    self.model_ava_bw_in_gpu = False
                    self.oh_res_pred_time += 1000 * (time.time() - time_0)
                    self.oh_res_pred_count += 1
                    if self.oh_res_pred_count % 100 == 0:
                        self.logger.info(f"overhead res_pred {self.oh_res_pred_time} {self.oh_res_pred_count}")
                if self.model_iter_time is not None and self.next_ava_cpu is not None and self.next_ava_bw is not None:
                    time_0 = time.time()
                    self.model_iter_time_in_gpu = True
                    self.next_iter_time = regr.predict(
                        self.model_iter_time, [self.next_ava_cpu, self.next_ava_bw], self.mean_list, self.std_list, self.device
                    )
                    self.model_iter_time_in_gpu = False
                    self.oh_itert_pred_time += 1000 * (time.time() - time_0)
                    self.oh_itert_pred_count += 1
                    if self.oh_itert_pred_count % 100 == 0:
                        self.logger.info(f"overhead itert_pred {self.oh_itert_pred_time} {self.oh_itert_pred_count}")
                if self.iter_count == 27000:
                    # self.ps_rrefs[0].rpc_sync().set_sync_mode(1)
                    rpc.rpc_sync(
                        to="manager",
                        func=set_use_heuristic,
                        args=(False,),
                    )

                rpc.rpc_sync(
                    to="manager",
                    func=set_next_iter_time,
                    args=(self.next_iter_time, self.wrk_id, self.epoch_idx),
                )

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
        model,
        worker_num,
        gpu_id,
        testing_data_dir,
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
        self.model = model
        self.worker_num = worker_num
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if model_name == "transformer":
            self.model = self.model.to(self.device)
        self.testing_data_dir = testing_data_dir
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

        if self.model_name == "lstm" or self.model_name == "transformer":
            train_iter = WikiText2(root=self.testing_data_dir, split="train")
            tokenizer = get_tokenizer("basic_english")
            vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
            vocab.set_default_index(vocab["<unk>"])
            ntokens = len(vocab)
            bptt = 35

            train_iter, val_iter, test_iter = WikiText2(root=self.testing_data_dir)
            val_data = data_process(val_iter, vocab, tokenizer)
            val_data = batchify(val_data, self.test_batch_size)

        time_init = time.time()
        time_0 = time_init
        # self.model = self.model.to(device)
        while True:
            time_1 = time.time()

            if time_1 - time_0 >= 40:
                time_0 = time_1

                if self.model_name == "transformer":
                    params = self.worker_rrefs[self.test_wrk_id].rpc_sync().get_model()
                    count = 0
                    for key, _ in self.model.named_parameters():
                        self.model.state_dict()[key].copy_(params[count])
                        count += 1
                else:
                    self.model = self.worker_rrefs[self.test_wrk_id].rpc_sync().get_model().to(self.device)

                test_correct = 0.0
                test_loss = 0.0

                if self.model_name == "lstm":
                    with torch.no_grad():
                        hidden = self.model.init_hidden(self.test_batch_size)
                        for _, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
                            data, targets = get_batch(val_data, i, bptt)
                            data, targets = data.to(self.device), targets.to(self.device)
                            hidden = repackage_hidden(hidden)
                            output, hidden = self.model(data, hidden)
                            loss = criterion(output, targets)
                            test_loss += len(data) * loss.item()
                elif self.model_name == "transformer":
                    with torch.no_grad():
                        for _, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
                            data, targets = get_batch(val_data, i, bptt)
                            data, targets = data.to(self.device), targets.to(self.device)
                            output = self.model(data)
                            output = output.view(-1, ntokens)
                            loss = criterion(output, targets)
                            test_loss += len(data) * loss.item()
                else:
                    with torch.no_grad():
                        for _, (data, target) in enumerate(self.test_data_loader):
                            data, target = data.to(self.device), target.to(self.device)
                            output = self.model(data)
                            loss = criterion(output, target)

                            test_loss += loss.item()
                            _, predicted = output.max(1)
                            test_correct += predicted.eq(target).sum().item()

                if self.model_name != "transformer":
                    self.model = self.model.cpu()

                if self.model_name == "lstm" or self.model_name == "transformer":
                    test_loss /= len(val_data) - 1
                else:
                    test_loss = test_loss * self.test_batch_size / len(self.test_data_loader.dataset)
                    test_accuracy = 100.0 * test_correct / len(self.test_dataset)

                step_num = rpc.rpc_sync(to="manager", func=get_step_num)

                if self.model_name == "lstm" or self.model_name == "transformer":
                    self.logger.info(
                        "Steps: {} | Loss: {:.4f} | Time: {:.4f} s".format(step_num, test_loss, time_1 - time_init)
                    )
                else:
                    self.logger.info(
                        "Steps: {} | Loss: {:.4f} | Acc.: {:.4f} % | Time: {:.4f} s".format(
                            step_num, test_loss, test_accuracy, time_1 - time_init
                        )
                    )

                if step_num >= 64000:
                    self.stop_flag = True
                    break

                # if test_loss <= self.test_target_loss:
                #     self.ps_rref.rpc_sync().stop_ps()
                #     break


wrk_num = 0
model_name = ""
batch_size = 0
model = None
wrk_rrefs = []
wrk_cm_times = []
wrk_cp_times = []
step_num = 0
step_num_lock = threading.Lock()
learning_rate = 0.0

# grad sync
future_grad_sync = torch.futures.Future()
future_grad_sync_count = 0
future_grad_sync_lock = threading.Lock()
ring_wrk_num = 0

# new update policy
policy_queue = queue.Queue()

child_wrk_rrefs = []


# ml part
use_heuristic = True
next_iter_times = []
next_iter_time_wrk_count = 0
next_iter_time_lock = threading.Lock()
sorted_next_iter_times = None

mngr_gpu_id = None
mngr_device = None
train_lstm_lock = threading.Lock()
train_reg_lock = threading.Lock()


# overhead
oh_heuristic_time = 0
oh_heuristic_count = 0
oh_ml_time = 0
oh_ml_count = 0
oh_wrk_wrk_pack_time = 0
oh_wrk_wrk_pack_count = 0


# epoch index
epoch_idx = 0


pgns = None

logger = None

sync_model = torch.load("./sync_model/sync_model.pth")
sync_meanlst = [0 for _ in range(13)]
sync_stdlst = [0 for _ in range(13)]

pre_straggler_num = 0

policy_dtm_period = 10
policy_dtm_period_count = 0


def set_use_heuristic(value):
    global use_heuristic
    use_heuristic = value


def determine_sync_policy():
    global sorted_next_iter_times
    global use_heuristic
    global oh_heuristic_time
    global oh_heuristic_count
    global oh_ml_time
    global oh_ml_count
    global oh_wrk_wrk_pack_time
    global oh_wrk_wrk_pack_count
    global epoch_idx
    global next_iter_times
    global wrk_num
    global pgns
    global batch_size
    global logger
    global model_name
    global step_num
    global learning_rate
    global sync_model
    global sync_meanlst
    global sync_stdlst
    global mngr_device
    global pre_straggler_num
    if use_heuristic:
        time_0 = time.time()
        epoch = epoch_idx
        if epoch > 9:
            epoch = 9
        iter_times = copy.deepcopy(next_iter_times)
        iter_times.sort()
        sync_num = 0
        for i in range(1, wrk_num):
            t1 = pgns.get_iter_num(int(i * batch_size), epoch) * iter_times[i - 1]
            t2 = pgns.get_iter_num(int((i + 1) * batch_size), epoch) * iter_times[i]
            if t1 < t2:
                sync_num = i
                break
        if sync_num == 0:
            sync_num == wrk_num
        pgns.set_cur_bsz(int(sync_num * batch_size))
    else:
        time_0 = time.time()
        input_data = []
        max_iter_time = sorted_next_iter_times[0][1]
        avg_iter_time = np.average(next_iter_times)
        straggling = (max_iter_time - avg_iter_time) / avg_iter_time
        input_data.append(straggling)
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
        input_data.extend(oneshot)
        input_data.append(learning_rate)
        input_data.append(step_num)
        value = regr.predict(sync_model, input_data, sync_meanlst, sync_stdlst, mngr_device)
        if value <= 0.0:
            sync_num = 0
        elif value >= wrk_num:
            sync_num = wrk_num
        else:
            value += 0.5
            value = int(value)
            sync_num = value

    # process sync num
    straggler_num = wrk_num - sync_num
    if straggler_num == wrk_num:
        straggler_num -= 1
    if straggler_num == pre_straggler_num:
        pass
    else:
        pre_straggler_num = straggler_num
        non_strgs = []
        strgs = []
        for i in range(wrk_num):
            wid = sorted_next_iter_times[i][0]
            if i < straggler_num:
                strgs.append(wid)
            else:
                non_strgs.append(wid)
        non_strgs.sort()
        strgs.sort()
        strg_dict = {}
        for idx in len(strgs):
            strg_id = strgs[idx]
            index = idx % len(non_strgs)
            strg_dict[strg_id] = index
        set_new_update_policy(non_strgs, strg_dict)

    if use_heuristic:
        oh_heuristic_time += 1000 * (time.time() - time_0)
        oh_heuristic_count += 1
        if oh_heuristic_count % 100 == 0:
            logger.info(f"overhead heuristic {oh_heuristic_time} {oh_heuristic_count}")
    else:
        oh_ml_time += 1000 * (time.time() - time_0)
        oh_ml_count += 1
        if oh_ml_count % 100 == 0:
            logger.info(f"overhead ml {oh_ml_time} {oh_ml_count}")


def set_next_iter_time(value, wrk_id, epoch):
    global wrk_num
    global next_iter_time_lock
    global next_iter_times
    global next_iter_time_wrk_count
    global sorted_next_iter_times
    global epoch_idx
    global policy_dtm_period
    global policy_dtm_period_count
    epoch_idx = epoch
    with next_iter_time_lock:
        next_iter_times[wrk_id] = value
        next_iter_time_wrk_count += 1
        if next_iter_time_wrk_count >= wrk_num:
            next_iter_time_wrk_count = 0
            policy_dtm_period_count += 1
            for e in next_iter_times:
                if e is None:
                    return
            sorted_next_iter_times = []
            for i in range(wrk_num):
                sorted_next_iter_times.append((i, next_iter_times[i]))
            sorted_next_iter_times.sort(key=lambda x: x[1], reverse=True)

            # determine policy
            if policy_dtm_period_count >= policy_dtm_period:
                policy_dtm_period_count = 0
                determine_sync_policy()


def mngr_init_global_vars(wnum, wrrefs, mdl, manager_gpu_id, mname, bsize, job_name, lr):
    global wrk_num
    global model
    global wrk_rrefs
    global wrk_cm_times
    global wrk_cp_times
    global ring_wrk_num
    global next_iter_times
    global mngr_gpu_id
    global mngr_device
    global pgns
    global batch_size
    global logger
    global model_name
    global learning_rate
    learning_rate = lr
    logger = Logger(job_name=job_name, file_path=f"./training_logs/{job_name}_manager.log").logger
    batch_size = bsize
    wrk_num = wnum
    model = mdl
    wrk_rrefs = wrrefs
    wrk_cm_times = [0.0 for _ in range(wnum)]
    wrk_cp_times = [0.0 for _ in range(wnum)]
    ring_wrk_num = wnum
    next_iter_times = [0.0 for _ in range(wnum)]
    mngr_gpu_id = manager_gpu_id
    mngr_device = torch.device(f"cuda:{mngr_gpu_id}" if torch.cuda.is_available() else "cpu")
    pgns = PGNS(mname, int(wnum * bsize), "./gradient_validation/")
    model_name = mname


def get_new_wrk_pkey_lists_locs(ring_wrk_num):
    global model
    pkey_numel = {}
    wrk_param_nums = [0 for _ in range(ring_wrk_num)]
    wrk_pkey_lists = [[] for _ in range(ring_wrk_num)]
    wrk_pkey_locs = {}
    for key, param in model.named_parameters():
        pkey_numel[key] = param.numel()
    pkey_numel = sorted(pkey_numel.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(pkey_numel)):
        key = pkey_numel[i][0]
        idx = np.argmin(wrk_param_nums)
        # param_ps_idx[key] = idx
        # param_loc_idx[key] = len(ps_param_lists[idx])
        wrk_param_nums[idx] += pkey_numel[i][1]
        wrk_pkey_locs[key] = (idx, len(wrk_pkey_lists[idx]))
        wrk_pkey_lists[idx].append(key)
    return wrk_pkey_lists, wrk_pkey_locs


def _check_and_change_update_policy():
    global wrk_rrefs
    global policy_queue
    global ring_wrk_num
    global child_wrk_rrefs
    try:
        policy = policy_queue.get(block=False)
    except Exception:
        return
    ring_wrk_ids = policy[0]
    child_parent: dict = policy[1]
    ftrs = []
    for cwref in child_wrk_rrefs:
        ftr = cwref.rpc_async().reset_child_wrk()
        ftrs.append(ftr)
    for ftr in ftrs:
        ftr.wait()
    # update wrk info
    futures = []
    ring_wrk_ids = list(ring_wrk_ids)
    ring_wrk_ids.sort()
    ring_wrk_num = len(ring_wrk_ids)
    wrk_pkey_lists, wrk_pkey_locs = get_new_wrk_pkey_lists_locs(ring_wrk_num)
    for i in range(ring_wrk_num):
        wref = wrk_rrefs[ring_wrk_ids[i]]
        pre_wrk_rref = wrk_rrefs[ring_wrk_ids[(i - 1) % ring_wrk_num]]
        ftr = wref.rpc_async().update_wrk_sync_info(True, 0, ring_wrk_num, i, pre_wrk_rref, wrk_pkey_lists, wrk_pkey_locs)
        futures.append(ftr)
    for k, e in child_parent.items():
        wref = wrk_rrefs[k]
        ftr = wref.rpc_async().update_wrk_sync_info(False, e, ring_wrk_num, 0, 0, wrk_pkey_lists, wrk_pkey_locs)
        futures.append(ftr)
    for ftr in futures:
        ftr.wait()
    for cwref in child_wrk_rrefs:
        cwref.rpc_async().let_child_wrk_use_new_policy()
    child_wrk_rrefs = []
    for k in child_parent.keys():
        child_wrk_rrefs.append(wrk_rrefs[k])


@rpc.functions.async_execution
def update_wrk_info(wrk_id, wrk_cm_t, wrk_cp_t, is_ring_wrk, added_step_num=1):
    global wrk_rrefs
    global step_num
    global step_num_lock
    global future_grad_sync
    global future_grad_sync_count
    global future_grad_sync_lock
    global ring_wrk_num
    global learning_rate
    wrk_cm_times[wrk_id] = wrk_cm_t
    wrk_cp_times[wrk_id] = wrk_cp_t
    with step_num_lock:
        step_num += added_step_num
        if step_num == 32000:
            futures = []
            learning_rate = 0.01
            for wref in wrk_rrefs:
                ftr = wref.rpc_async().update_optimizer(0.01)
                futures.append(ftr)
            for ftr in futures:
                ftr.wait()
        elif step_num == 48000:
            futures = []
            learning_rate = 0.001
            for wref in wrk_rrefs:
                ftr = wref.rpc_async().update_optimizer(0.001)
                futures.append(ftr)
            for ftr in futures:
                ftr.wait()
    if is_ring_wrk:
        ftr = future_grad_sync
        with future_grad_sync_lock:
            future_grad_sync_count += 1
            if future_grad_sync_count >= ring_wrk_num:
                future_grad_sync_count = 0
                _check_and_change_update_policy()
                ftr.set_result(0)
                future_grad_sync = torch.futures.Future()
        return ftr
    else:
        ftr = torch.futures.Future()
        ftr.set_result(0)
        return ftr


def get_step_num():
    global step_num
    return step_num


def set_new_update_policy(ring_wrk_ids, child_parent: dict):
    global policy_queue
    policy_queue.put((ring_wrk_ids, child_parent))


def train_sr_lstm_model(seq_data, model=None):
    with train_lstm_lock:
        if model is None:
            model = sr_lstm.ShallowRegressionLSTM(num_features=1, hidden_units=32)
        model.set_device(mngr_device)
        model = model.to(mngr_device)
        model.train()
        loss_fn = nn.MSELoss().to(mngr_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        dataset = sr_lstm.SequenceDataset(seq_data, 20)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
        for epc in range(1000):
            # print(f"Epoch {epc}\n---------")
            sr_lstm.train_model(data_loader, model, loss_fn, optimizer, mngr_device)

        device = torch.device("cpu")
        model.set_device(device)
        model = model.to(device)

    return model


def train_reg_model(input, label, model=None):
    with train_reg_lock:
        model, mean_lst, std_lst = regr.train_model(input, label, 10, mngr_device, model)

        device = torch.device("cpu")
        # model.set_device(device)
        model = model.to(device)

    return (model, mean_lst, std_lst)


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
    wrk_server_ids,
    wrk_pack_toggles,
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
        elif model_name == "lstm":
            model = lstm.RNNModel(rnn_type="LSTM", ntoken=28782, ninp=200, nhid=200, nlayers=2, dropout=0)
        elif model_name == "transformer":
            model = transformer.TransformerModel(ntoken=28782, ninp=200, nhead=8, nhid=200, nlayers=2, dropout=0)
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
                    gpu_ids[id + 1],
                    wrk_server_ids[id],
                    wrk_pack_toggles[id],
                    training_data_dir,
                    batch_size,
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
                model,
                wrk_num,
                gpu_ids[-1],
                training_data_dir,
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

        mngr_init_global_vars(wrk_num, wrk_rrefs, model, gpu_ids[0], model_name, batch_size, job_name, learning_rate)

        futures = []
        for id in range(wrk_num):
            futures.append(rpc.rpc_async(to=f"worker{id}", func=Worker.run_worker, args=(wrk_rrefs[id],)))
        futures.append(rpc.rpc_async(to="tester", func=Tester.test_model, args=(tester_rref,)))

        # time.sleep(10)
        # set_new_update_policy([0, 1, 2], {3: 1})
        # time.sleep(20)
        # set_new_update_policy([0, 3], {2: 0, 1: 3})

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
    parser.add_argument("--wrk_server_ids", type=str, required=True)
    parser.add_argument("--wrk_pack_toggles", type=str, required=True)
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
        args.wrk_server_ids.split(","),
        args.wrk_pack_toggles.split(","),
        # args.model_fetch_ids.split(","),
    )
