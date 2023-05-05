#! /usr/bin/env python3

import argparse
import socket
import threading
import time

import gpustat
import psutil

# from res_limiter import run_res_limiter
from zeyu_utils import net as znet
from zeyu_utils import os as zos

res_info = None


def get_cpu_memory_gpu():
    cpu_usage = psutil.cpu_percent(interval=1)

    virtual_memory = psutil.virtual_memory()
    memory_usage = virtual_memory.used / 1024 / 1024 / 1024

    try:
        query = gpustat.new_query()
    except Exception:
        gpu_memory = 0
        gpu_util = 0
    else:
        gpu_memory = 0
        gpu_util = 0
        for gpu in query.gpus:
            gpu_memory += 100.0 * gpu.memory_used / gpu.memory_total
            gpu_util += gpu.utilization

    return cpu_usage, memory_usage, gpu_memory, gpu_util


def update_res_info():
    global res_info

    memory_usage = psutil.virtual_memory().used / 1024 / 1024 / 1024
    bw = psutil.net_io_counters(pernic=True)
    # nic = socket.if_nameindex()[1][1]
    nic = "eth0"
    temp_recv = bw[nic].bytes_recv
    temp_sent = bw[nic].bytes_sent

    t0 = time.time()
    while True:
        t1 = time.time()
        if t1 - t0 > 1:
            t0 = t1
            bw = psutil.net_io_counters(pernic=True)
            cpu_usage, memory_usage, gpu_memory, gpu_util = get_cpu_memory_gpu()

            recv_rate = (bw[nic].bytes_recv - temp_recv) / 1024.0 / 1024.0
            send_rate = (bw[nic].bytes_sent - temp_sent) / 1024.0 / 1024.0

            res_info = f"{cpu_usage:6.2f} {memory_usage:5.2f} {gpu_memory:3.0f} {gpu_util:3d} {recv_rate:7.2f} {send_rate:7.2f}"

            temp_recv = bw[nic].bytes_recv
            temp_sent = bw[nic].bytes_sent


def res_info_conn_thread(connm: znet.SocketMsger):
    while True:
        entity = connm.recv()
        if entity is None or entity == "":
            return
        # print(f"Respond {entity}")
        connm.send(res_info)


def res_info_listener_thread():
    listener = znet.SocketMsger.tcp_listener("0.0.0.0", 9093)
    while True:
        connm, _ = listener.accept()
        threading.Thread(target=res_info_conn_thread, args=(connm,)).start()


ps_connms = {}


def ps_conn_thread(connm: znet.SocketMsger):
    global ps_connms
    job_name = connm.recv()
    ps_connms[job_name] = connm


def ps_listener_thread():
    listener = znet.SocketMsger.tcp_listener("0.0.0.0", 55673)
    while True:
        connm, _ = listener.accept()
        threading.Thread(target=ps_conn_thread, args=(connm,)).start()


def dmaker_conn_thread(connm: znet.SocketMsger):
    global ps_connms
    ps_connm = None
    job_name = connm.recv()
    while True:
        if job_name in ps_connms:
            ps_connm = ps_connms[job_name]
            break
        time.sleep(1)
    while True:
        data = connm.recv()
        if data is None or (isinstance(data, str) and data == ""):
            ps_connm.close()
            ps_connms.pop(job_name)
            return
        if data == "G":
            ps_connm.send("G")
            ps_data = ps_connm.recv()
            connm.send(ps_data)
        else:
            ps_connm.send(data)


def dmaker_listener_thread():
    listener = znet.SocketMsger.tcp_listener("0.0.0.0", 46996)
    while True:
        connm, _ = listener.accept()
        threading.Thread(target=dmaker_conn_thread, args=(connm,)).start()


def main():
    t0 = threading.Thread(target=update_res_info)
    t1 = threading.Thread(target=res_info_listener_thread)
    t2 = threading.Thread(target=ps_listener_thread)
    t3 = threading.Thread(target=dmaker_listener_thread)
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()


if __name__ == "__main__":
    # threading.Thread(target=run_res_limiter).start()
    main()
