#! /usr/bin/env python3


import argparse
import logging
import time
from multiprocessing import Process

import psutil
from zeyu_utils import net as znet
from zeyu_utils import os as zos

server_nic = "ens3"
time_interval = 1


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


def start_a_new_recording_process(
    for_server=True, res_type="cpu", server_name="server0", task_name="job0_resnet20_ps", model_name=""
):
    if for_server:
        logger = Logger(job_name=f"{server_name}_{res_type}", file_path=f"./training_logs/{model_name}_{res_type}.log").logger
        if res_type == "cpu":
            while True:
                cpup = psutil.cpu_percent(interval=time_interval)
                logger.info(f"{cpup}")
        elif res_type == "bw":
            sent0 = psutil.net_io_counters(pernic=True)[server_nic].bytes_sent
            recv0 = psutil.net_io_counters(pernic=True)[server_nic].bytes_recv
            time0 = time.time()
            while True:
                time.sleep(time_interval)
                sent1 = psutil.net_io_counters(pernic=True)[server_nic].bytes_sent
                recv1 = psutil.net_io_counters(pernic=True)[server_nic].bytes_recv
                time1 = time.time()
                time_diff = time1 - time0
                in_bw = (recv1 - recv0) / 1048576 / time_diff
                out_bw = (sent1 - sent0) / 1048576 / time_diff
                logger.info(f"in/out {in_bw:.4f} {out_bw:.4f}")
                sent0 = sent1
                recv0 = recv1
                time0 = time1
    else:
        logger = Logger(job_name=f"{task_name}_{res_type}", file_path=f"./training_logs/{task_name}_{res_type}.log").logger
        if res_type == "cpu":
            jname, mname, task = tuple(task_name.split("_"))
            if task == "ps":
                rpc_rank = 1
            elif "wrk" in task:
                wid = int(task[-1])
                rpc_rank = 2 + wid

            key_str = f"\\-\\-job_name={jname} \\-\\-model_name={mname} \\-\\-rpc_rank={rpc_rank}"
            cmd = f"ps aux | grep '{key_str}' | grep -v grep | awk '{{print $2}}'"
            pid = zos.run_cmd(cmd)
            if "\n" in pid:
                pid = pid.split("\n")[0]
            p = psutil.Process(pid=int(pid))

            while True:
                cpup = p.cpu_percent(interval=time_interval)
                logger.info(f"{cpup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--server_name", type=str, default="server0")
    parser.add_argument("--task_names", type=str, default="job0_resnet20_ps,job0_resnet20_wrk0")
    parser.add_argument("--model_name", type=str, default="resnet20")

    args = parser.parse_args()
    server_name = args.server_name
    task_names = args.task_names
    task_names = task_names.split(",")

    server_p_cpu = Process(
        target=start_a_new_recording_process,
        args=(True, "cpu", server_name, "", args.model_name),
    )
    server_p_cpu.start()

    server_p_bw = Process(
        target=start_a_new_recording_process,
        args=(True, "bw", server_name, "", args.model_name),
    )
    server_p_bw.start()

    # for tsk_name in task_names:
    #     Process(
    #         target=start_a_new_recording_process,
    #         args=(False, "cpu", server_name, tsk_name),
    #     ).start()

    server_p_cpu.join()
