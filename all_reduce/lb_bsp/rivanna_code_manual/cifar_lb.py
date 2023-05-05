import argparse
import logging
import os
import pickle
import random
import socket
import sys
import threading
import time
from datetime import datetime

import GPUtil
import numpy as np
import pandas as pd
import psutil
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from model import alexnet, densenet, googlenet, mobilenetv2, resnet3, vgg
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

res_limiter_port = 43163

rand_seed = 123
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



class SocketMsger:
    def __init__(self, socket, is_listener=False):
        self.piggyback_data = None
        self.__socket = socket
        self.__is_listener = is_listener
        self.__is_blocking = True
        self.__recv_buffer = b""
        self.__closed = False

    @property
    def socket(self):
        return self.__socket

    @property
    def is_listener(self):
        return self.__is_listener

    @property
    def is_blocking(self):
        return self.__is_blocking

    @property
    def closed(self):
        if getattr(self.__socket, "_closed") is True and self.__closed is False:
            self.__closed = True
        return self.__closed

    def send(self, data):
        if self.__closed or self.__is_listener:
            return
        if isinstance(data, str):
            data_type = 0
            byte_data = data.encode()
        elif isinstance(data, bytes):
            data_type = 1
            byte_data = data
        else:
            data_type = 2
            byte_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        data_length = len(byte_data)
        self.__socket.sendall(f"META({data_type},{data_length})".encode() + byte_data)

    def recv(self, blocking=True):
        if self.__closed or self.__is_listener:
            return
        if blocking:
            if not self.__is_blocking:
                self.__socket.setblocking(True)
                self.__is_blocking = True
        else:
            if self.__is_blocking:
                self.__socket.setblocking(False)
                self.__is_blocking = False
        index = self.__recv_buffer.find(b"META(")
        while index == -1:
            try:
                data = self.__socket.recv(1024)
                if data == b"":
                    self.__closed = True
                    return
                self.__recv_buffer += data
                index = self.__recv_buffer.find(b"META(")
            except BlockingIOError:
                return
        meta_lindex = index + 5
        index = self.__recv_buffer.find(b")", meta_lindex)
        while index == -1:
            try:
                data = self.__socket.recv(1024)
                if data == b"":
                    self.__closed = True
                    return
                self.__recv_buffer += data
                index = self.__recv_buffer.find(b")", meta_lindex)
            except BlockingIOError:
                return
        meta_rindex = index
        meta = self.__recv_buffer[meta_lindex:meta_rindex].split(b",")
        data_type = int(meta[0])
        data_length = int(meta[1])
        body_lindex = meta_rindex + 1
        while len(self.__recv_buffer) - body_lindex < data_length:
            try:
                data = self.__socket.recv(1024)
                if data == b"":
                    self.__closed = True
                    return
                self.__recv_buffer += data
            except BlockingIOError:
                return
        body_rindex = body_lindex + data_length
        recvd_data = self.__recv_buffer[body_lindex:body_rindex]
        self.__recv_buffer = self.__recv_buffer[body_rindex:]
        if data_type == 0:
            return recvd_data.decode()
        elif data_type == 1:
            return recvd_data
        else:
            return pickle.loads(recvd_data)

    def close(self):
        self.__socket.close()
        self.__closed = True

    @staticmethod
    def tcp_listener(listening_ip, listening_port, backlog=100):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listening_ip, listening_port))
        listener.listen(backlog)
        return SocketMsger(listener, True)

    def accept(self):
        if self.__is_listener:
            conn, address = self.__socket.accept()
            connm = SocketMsger(conn)
            return connm, address

    @staticmethod
    def tcp_connect(ip, port, retry=True):
        sock = None
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((ip, port))
                return SocketMsger(sock)
            except Exception as e:
                print("NOT CONNECTED:", e, file=sys.stderr)
                if not retry:
                    return
                time.sleep(1)


class Logger(object):
    def __init__(self, job_name, file_dir, log_level=logging.INFO, mode="w"):
        self.logger = logging.getLogger(job_name)
        self.logger.setLevel(log_level)
        self.fh = logging.FileHandler(filename=file_dir, mode=mode)
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)


class ParameterServer(object):
    def __init__(self, model, num_workers, lr, job_name):
        self.lock = threading.Lock()
        self.logger = Logger(job_name=job_name, file_dir=f"./logs/{job_name}_ps.log").logger
        self.job_name = job_name
        # self.cm_t1_start = np.zeros(num_workers)
        self.future_model = torch.futures.Future()
        self.batch_update_size = num_workers
        self.curr_update_size = 0
        self.stop_flag = False
        self.all_worker_info = {"worker1": [0, 0, 0.0], "worker2": [0, 0, 0.0], "worker3": [0, 0, 0.0], "worker4": [0, 0, 0.0]}
        # {worker1: [worker-to-ps, ps-to-worker]}
        self.all_comm_time = {"worker1": [0.0, 0.0], "worker2": [0.0, 0.0], "worker3": [0.0, 0.0], "worker4": [0.0, 0.0]}
        self.ps_update_time = 0.0
        self.sync_mode = "0"
        if model == "resnet20":
            self.model = resnet3.resnet20()
        elif model == "resnet56":
            self.model = resnet3.resnet56()
        elif model == "vgg13":
            self.model = vgg.VGG13()
        elif model == "vgg16":
            self.model = vgg.VGG16()
        elif model == "densenet121":
            self.model = densenet.DenseNet121()
        elif model == "alexnet":
            self.model = alexnet.AlexNet()
        elif model == "googlenet":
            self.model = googlenet.GoogLeNet()
        elif model == "mobilenet":
            self.model = mobilenetv2.MobileNetV2()
        self.lr = lr
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def set_worker_info(self, worker_name, info):
        self.all_worker_info[worker_name] = info

    def set_comm_time(self, worker_name, comm_time):
        self.all_comm_time[worker_name] = comm_time

    def get_model(self):
        return self.model

    def stop(self):
        self.stop_flag = True

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, name, i, batch_idx, cm_t0_start, cm_t1_start, cm_t1_end):
        self: ParameterServer = ps_rref.local_value()
        cm_t0_end = time.time()
        cm_t0 = 1000 * (cm_t0_end - cm_t0_start)
        worker_rank = int(name[-1])
        cm_t1 = 1000 * (cm_t1_end - cm_t1_start)

        self.logger.info(
            "Epoch: {:3d} | Batch: {:3d} | {:8s} | Communication O: {:7.2f} ms".format((i + 1), batch_idx - 1, name, cm_t1)
        )
        self.logger.info(
            "Epoch: {:3d} | Batch: {:3d} | {:8s} | Communication I: {:7.2f} ms".format((i + 1), batch_idx, name, cm_t0)
        )

        self.set_comm_time(f"worker{worker_rank}", [cm_t0, cm_t1])

        data =[]
        data.append(datetime.now())
        data.append(i)
        data.append(cm_t0)
        data.append(cm_t1)
        data.append(cm_t0/1000)
        data.append(cm_t1/1000)
        # data.append(cpu_use)
        # data.append(memoryUse)
        # data.append(cp_t)
        # gpu_index_count = 0
        # for each_gpu in gpus:
        #     data.append(each_gpu.load)
        #     data.append(each_gpu.memoryUsed)

        data_csv = []
        data_csv.append(data)
        import csv
        file_common = './logs/measurement_cm_'+str(self.job_name)+'worker'+str(worker_rank)+'.csv'
        with open(file_common, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_csv)

        with self.lock:
            sync_mode = self.sync_mode
            for p, g in zip(self.model.parameters(), grads):
                p.grad += g
            if sync_mode == "0":
                self.curr_update_size += 1
            fut = self.future_model

            if sync_mode == "1" or self.curr_update_size >= self.batch_update_size:
                # with torch.no_grad():
                # for p in self.model.parameters():
                # p.grad /= self.batch_update_size
                # p += -self.lr * p.grad
                # p.grad = torch.zeros_like(p)
                update_time_0 = time.time()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if sync_mode == "0":
                    self.curr_update_size = 0
                update_time_1 = time.time()

                self.logger.info("Update Time: {:.2f} ms".format(1000 * (update_time_1 - update_time_0)))
                self.ps_update_time = 1000 * (update_time_1 - update_time_0)

                t = time.time()

                self.logger.info(f"PS sending updated parameters to {name}")

                fut.set_result([self.model, self.stop_flag, t])
                self.future_model = torch.futures.Future()

        return fut


def run_worker(ps_rref, data_dir, batch_size, num_epochs, worker, job_name, model, gpu_id):
    logger = Logger(job_name=job_name, file_dir=f"./logs/{job_name}_{worker}.log").logger

    pid = os.getpid()
    current_process = psutil.Process(pid)
    # current_process.cpu_affinity([0])


    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device_id = gpu_id
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    name = rpc.get_worker_info().name

    m = ps_rref.rpc_sync().get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    stop_flag = False

    # Connect to res_limiter
    if worker == "worker3":
        res_limiter_connm = SocketMsger.tcp_connect("127.0.0.1", res_limiter_port)
        res_limiter_connm.send(job_name)

    cm_t1_end = time.time()
    tt0 = time.time()

    cm_t1_start = 0.0
    for i in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()

            if batch_idx > 5:
                file_read = './logs/'+str(job_name)+'data.csv'
                df = pd.read_csv(file_read, header=None)

                index_worker_all = df[0][len(df)-4: len(df)]

                cp_t_all = df[1][len(df)-4: len(df)]

                # print(index_worker_all)
                # print(cp_t_all)


                min_time = min(cp_t_all)
                # print(min_time)

                max_time = max(cp_t_all)
                # print(max_time)

                # worker_index = 0
                if max_time/min_time > 1.5:
                    for item1, item2 in zip(cp_t_all, index_worker_all):
                        # print(item1)
                        # print("\n")
                        # print(item2)

                        if max_time == item1 and int(worker[6:len(worker)]) == item2:
                            # computation_straggler = True
                            if batch_size > 10:
                                batch_size = batch_size - 5
                                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                data, target = next(iter(train_loader))
                                break

                        elif min_time == item1 and int(worker[6:len(worker)]) == item2:
                            # computation_straggler = True
                            # if batch_size < 250:
                            batch_size = batch_size + 5
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            data, target = next(iter(train_loader))
                            break





            data, target = data.to(device), target.to(device)
            output = m(data)
            loss = criterion(output, target)
            loss.backward()

            cpu_use=current_process.cpu_percent(interval=None)
            memoryUse = current_process.memory_info()[0]/2.**20
            io_counters = current_process.io_counters()
            disk_usage_process = (io_counters[2] + io_counters[3])/(1024*1024)

            cm_t0_start = time.time()
            cp_t = 1000 * (cm_t0_start - cm_t1_end)


            gpus = GPUtil.getGPUs()

            data =[]
            data.append(datetime.now())
            data.append(i)
            data.append(cpu_use)
            data.append(memoryUse)
            data.append(disk_usage_process)
            data.append(cp_t)
            data.append(cp_t/1000)
            gpu_index_count = 0
            for each_gpu in gpus:
                data.append(each_gpu.load)
                data.append(each_gpu.memoryUsed)

            # data_csv = []
            # data_csv.append(data)
            # import csv
            # file_common = 'Archive/logs_lb/measurement_cp_'+str(job_name)+'worker'+str(worker[6:len(worker)])+'.csv'
            # with open(file_common, 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerows(data_csv)

            logger.info(
                "{:8s} | Epoch: {:3d} | Batch: {:3d} | Loss: {:6.2f} | Computation Time: {:7.2f} ms".format(
                    name, (i + 1), (batch_idx + 1), loss.item(), cp_t
                )
            )

            ps_rref.rpc_async().set_worker_info(worker, [(i + 1), (batch_idx + 1), cp_t])

            m, stop_flag, t = rpc.rpc_sync(
                to=ps_rref.owner(),
                func=ParameterServer.update_and_fetch_model,
                args=(ps_rref, [p.grad for p in m.cpu().parameters()], name, i, batch_idx, cm_t0_start, cm_t1_start, cm_t1_end),
            )
            cm_t1_start = t
            m.to(device)


            cm_t1_end = time.time()

            end_time =time.time()
            it = end_time - start_time
            data.append(it)

            data_csv = []
            data_csv.append(data)
            import csv
            file_common = './logs/measurement_cp_'+str(job_name)+'worker'+str(worker[6:len(worker)])+'.csv'
            with open(file_common, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data_csv)

            c_data = []
            c_data.append(int(worker[6:len(worker)]))
            c_data.append(cp_t+(cm_t1_end - cm_t1_start))
            # c_data.append(examples_per_sec)
            # data.append(train_accuracy.result())
            # data.append(test_accuracy.result())
            # c_data.append(split_2[0])
            # c_data.append(split_2_2[0])

            c_data_csv = []
            c_data_csv.append(c_data)

            import csv
            c_file_data = './logs/'+str(job_name)+'data.csv'

            with open(c_file_data, 'a') as file:
                wr2_c = csv.writer(file)
                wr2_c.writerows(c_data_csv)
            # else:
            #     with open(c_file_data, 'a', newline ='') as file:
            #         wr2_c = csv.writer(file)
            #         wr2_c.writerows(c_data_csv)


            if stop_flag:
                break

        if stop_flag:
            break

    tt1 = time.time()

    ###########job completion status logger

    if worker == "worker4":
        common_data = []
        common_data.append(int(job_name[3:len(job_name)]))
        common_data.append(worker)
        common_data.append(1)
        # common_data.append(gs)
        # common_data.append(loss_value)
        # common_data.append(duration)
        # common_data.append(accuracy)
        # common_data.append(val_accuracy)

        common_data_csv = []
        common_data_csv.append(common_data)
        import csv
        file_common = './complete_status_lb.csv'
        with open(file_common, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(common_data_csv)

    if worker == "worker3":
        res_limiter_connm.close()

    logger.info("Time: {:.2f} seconds".format((tt1 - tt0)))


def get_accuracy(ps_rref, data_dir, test_batch_size, job_name, target_loss, model, gpu_id):
    logger = Logger(job_name=job_name, file_dir=f"./logs/{job_name}_tester.log").logger
    logger.info(f"model_name: {model}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    dataset_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    t0 = time.time()
    logger.info("Start!")
    init = t0
    while True:
        t1 = time.time()
        if t1 - t0 > 40:
            t0 = t1
            m = ps_rref.rpc_sync().get_model().to(device)

            test_loss = 0
            correct = 0

            with torch.no_grad():
                for j, (data, target) in enumerate(test_loader):
                    data, target = data.to(device), target.to(device)
                    output = m(data)
                    loss = criterion(output, target)

                    test_loss += loss.item()
                    _, predicted = output.max(1)
                    correct += predicted.eq(target).sum().item()

            test_loss = test_loss * test_batch_size / len(test_loader.dataset)
            accuracy = 100.0 * correct / len(dataset_test)
            logger.info(
                "Test Loss: {:6.3f} | Accuracy: {:5.2f} % | Time: {:7.2f} seconds".format(test_loss, accuracy, (t1 - init))
            )

            if test_loss < target_loss:
                ps_rref.rpc_sync().stop()
                break
            # elif (model == "alexnet" and (t1 - init) > 1800) or ((t1 - init) > 3600):
            #     ps_rref.rpc_sync().stop()
            #     logger.info("Time_is_out!")
            #     break


def report_info_ps_thread(ps: ParameterServer):
    connm = SocketMsger.tcp_connect("172.17.0.1", 55673)
    connm.send(ps.job_name)
    while True:
        data = connm.recv()
        if (ps.stop_flag is True) or (data is None) or (isinstance(data, str) and data == ""):
            return
        if data == "G":
            connm.send((ps.all_worker_info, ps.all_comm_time, ps.ps_update_time))
        else:
            ps.sync_mode = data


def run(rank, num_workers, data_dir, model, batch_size, test_batch_size, lr, num_epochs, job_name, target_loss, gpu_id):
    logging.basicConfig(level=logging.INFO)
    world_size = num_workers + 2
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=0)

    if rank == 0:
        logging.info(f"PS{rank} initializing")
        rpc.init_rpc(f"PS{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
        logging.info(f"PS{rank} initialized")

        workers = [f"worker{r}" for r in range(1, world_size - 1)]
        ps = ParameterServer(model, num_workers, lr, job_name)
        ps_rref = rpc.RRef(ps)

        # threading.Thread(target=report_info_ps_thread, args=(ps,)).start()

        futs = []
        futs.append(
            rpc.rpc_async(
                to="tester", func=get_accuracy, args=(ps_rref, data_dir, test_batch_size, job_name, target_loss, model, gpu_id)
            )
        )
        for worker in workers:
            futs.append(
                rpc.rpc_async(
                    to=worker, func=run_worker, args=(ps_rref, data_dir, batch_size, num_epochs, worker, job_name, model, gpu_id)
                )
            )

        torch.futures.wait_all(futs)
        logging.info(f"Finish training")

    elif rank == world_size - 1:
        logging.info(f"Tester initializing")
        rpc.init_rpc("tester", rank=rank, world_size=world_size, rpc_backend_options=options)
        logging.info(f"Tester initialized")

    else:
        logging.info(f"Worker{rank} initializing")
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
        logging.info(f"Worker{rank} initialized")

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on Imagenet under SSGD")
    parser.add_argument("--job_name", type=str, default="test", help="The job's name.")
    parser.add_argument("--model", type=str, default="resnet50", help="The model's name.")
    parser.add_argument("--rank", type=int, default=1, help="Global rank of this process.")
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of workers.")
    parser.add_argument("--data_dir", type=str, default="./data", help="The location of dataset.")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Address of master.")
    parser.add_argument("--master_port", type=str, default="29600", help="Port that master is listening on.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of each worker during training.")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Batch size during testing.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--target_loss", type=float, default=0.8, help="Targer accuracy.")  # 0.8
    parser.add_argument("--gpu_id", type=int, default=0, help="Targer accuracy.")  # 0.8


    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    run(
        args.rank,
        args.num_workers,
        args.data_dir,
        args.model,
        args.batch_size,
        args.test_batch_size,
        args.lr,
        args.num_epochs,
        args.job_name,
        args.target_loss,
        args.gpu_id
    )
