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


model = None
wrk_rrefs = []
wrk_cm_times = []
wrk_cp_times = []
step_num = 0
step_num_lock = threading.Lock()

# grad sync
future_grad_sync = torch.futures.Future()
future_grad_sync_count = 0
future_grad_sync_lock = threading.Lock()
ring_wrk_num = 0

# new update policy
policy_queue = queue.Queue()

child_wrk_rrefs = []


def mngr_init_global_vars(wrk_num, wrrefs, mdl):
    global model
    global wrk_rrefs
    global wrk_cm_times
    global wrk_cp_times
    global ring_wrk_num
    model = mdl
    wrk_rrefs = wrrefs
    wrk_cm_times = [0.0 for _ in range(wrk_num)]
    wrk_cp_times = [0.0 for _ in range(wrk_num)]
    ring_wrk_num = wrk_num


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
    wrk_cm_times[wrk_id] = wrk_cm_t
    wrk_cp_times[wrk_id] = wrk_cp_t
    with step_num_lock:
        step_num += added_step_num
        if step_num == 32000:
            futures = []
            for wref in wrk_rrefs:
                ftr = wref.rpc_async().update_optimizer(0.01)
                futures.append(ftr)
            for ftr in futures:
                ftr.wait()
        elif step_num == 48000:
            futures = []
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
    init_sync_policy,
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
                    gpu_ids[id],
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

        mngr_init_global_vars(wrk_num, wrk_rrefs, model)

        print(init_sync_policy)
        if init_sync_policy == "None":
            pass
        else:
            ring = init_sync_policy.split("S")[0].split(",")
            ring = list(map(int, ring))
            dictionary = {}
            child_parent = init_sync_policy.split("S")[1].split(",")
            for e in child_parent:
                child = int(e.split("C")[0])
                parent = int(e.split("C")[1])
                dictionary[child] = parent
            set_new_update_policy(ring, dictionary)
        # set_new_update_policy([0, 2, 4, 6], {1: 0, 3: 2, 5: 4, 7: 6})
        # set_new_update_policy([0], {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0})
        # set_new_update_policy([0, 1, 2, 3], {4: 0, 5: 1, 6: 2, 7: 3})

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
    parser.add_argument("--init_sync_policy", type=str, default="None")
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
        args.init_sync_policy,
        # args.model_fetch_ids.split(","),
    )
