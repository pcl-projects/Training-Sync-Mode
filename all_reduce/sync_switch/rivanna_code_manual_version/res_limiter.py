#! /usr/bin/env python3


import threading
import time

from zeyu_utils import net as znet
from zeyu_utils import os as zos

# import argparse


res_limiter_port = 43163
# notifier_port = 43164
cpulimit_path = "/home/qxc4fh/zeyu_workspace/bin/cpulimit"
limit_percent = "10"
limit_time_interval = 60

# job_notifiers = {}


def limit_job(job_pid):
    limit_cmd = f"{cpulimit_path} -p {job_pid} -l {limit_percent} -z"
    threading.Thread(target=zos.run_cmd, args=(limit_cmd,)).start()
    print(f"Limit job with PID {job_pid}")


def unlimit_job(job_pid):
    key_str = f"\\-p {job_pid} \\-l {limit_percent} \\-z"
    cmd = f"ps aux | grep '{key_str}' | grep -v grep | awk '{{print $2}}'"
    limit_cmd_pid = zos.run_cmd(cmd)
    zos.run_cmd(f"kill {limit_cmd_pid}")
    print(f"Unlimit job with PID {job_pid}")


def conn_thread(connm: znet.SocketMsger):
    job_name = connm.recv()
    key_str = f"\\-\\-job_name={job_name}"
    cmd = f"ps aux | grep '{key_str}' | grep cifar | grep -v grep | awk '{{print $2}}'"
    job_pid = zos.run_cmd(cmd)
    limit_flag = True
    while True:
        time.sleep(limit_time_interval)
        if connm.closed:
            # job_notifiers.pop(job_name)
            return
        if limit_flag is True:
            limit_flag = False
            limit_job(job_pid)
            # job_notifiers[job_name].send("0")
        else:
            limit_flag = True
            unlimit_job(job_pid)
            # job_notifiers[job_name].send("0")


def res_limiter_listener():
    listener = znet.SocketMsger.tcp_listener("0.0.0.0", res_limiter_port)
    while True:
        connm, _ = listener.accept()
        threading.Thread(target=conn_thread, args=(connm,)).start()


# def notifier_conn_thread(connm: znet.SocketMsger):
#     job_name = connm.recv()
#     job_notifiers[job_name] = connm


# def exp_testing_notifier():
#     listener = znet.SocketMsger.tcp_listener("0.0.0.0", notifier_port)
#     while True:
#         connm, _ = listener.accept()
#         threading.Thread(target=notifier_conn_thread, args=(connm,)).start()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--jn", type=str, default="test", help="The job's name.")
    # args = parser.parse_args()
    # key_str = f"--job_name={args.jn}"
    t1 = threading.Thread(target=res_limiter_listener)
    # t2 = threading.Thread(target=exp_testing_notifier)
    t1.start()
    # t2.start()
    t1.join()
    # t2.join()
