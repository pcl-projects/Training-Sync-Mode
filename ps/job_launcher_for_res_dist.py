#! /usr/bin/env python3


import logging
import threading
import time

import numpy as np
from blossom import find_maximum_matching
from zeyu_utils import net as znet

cmd_client_max_num = 4
server_ips = ["10.153.5.126", "10.153.2.120", "10.153.2.121", "10.153.5.23"]
server_ips = ["10.153.5.23", "10.153.2.121", "10.153.2.120", "10.153.5.126"]
res_monitor_ports = [39271, 39272, 39273, 39274]
job_rpc_master_port_init = 59807


workspace = "/home/ubuntu/sgd/"
workspace = "/home/qxc4fh/zeyu_workspace/sgd/"


listener_port_for_cmd = 30009


def run_remote_py(sockm, pyfname="training_sgd.py", path_prefix=workspace, **kwargs):
    job_name = kwargs["job_name"]
    model_name = kwargs["model_name"]
    rpc_rank = kwargs["rpc_rank"]
    ps_num = kwargs["ps_num"]
    worker_num = kwargs["worker_num"]
    rpc_master_addr = kwargs["rpc_master_addr"]
    rpc_master_port = kwargs["rpc_master_port"]
    epoch_num = kwargs["epoch_num"]
    data_partitioned = kwargs["data_partitioned"]
    gpu_ids = kwargs["gpu_ids"]
    ps_server_id = kwargs["ps_server_id"]
    wrk_server_ids = kwargs["wrk_server_ids"]
    wrk_pack_toggles = kwargs["wrk_pack_toggles"]
    tree_groups = kwargs["tree_groups"]
    lr = kwargs["lr"]

    cmd = f"python3 {path_prefix}/{pyfname} --job_name={job_name} --model_name={model_name} --rpc_rank={rpc_rank} --ps_num={ps_num} --worker_num={worker_num} --rpc_master_addr={rpc_master_addr} --rpc_master_port={rpc_master_port} --epoch_num={epoch_num} --data_partitioned={data_partitioned} --gpu_ids={gpu_ids} --wrk_server_ids={wrk_server_ids} --wrk_pack_toggles={wrk_pack_toggles} --tree_groups={tree_groups} --learning_rate={lr} --ps_server_id={ps_server_id}"

    sockm.send(cmd)


# Also exported for out-package use
# Old design
def ps_wrk_packing(jobs_data):
    # jobs_data: [(job_id, ps_idle, wrk_active)]
    ps_idle_times = []
    wrk_active_times = []
    for e in jobs_data:
        ps_idle_times.append((e[0], e[1]))
        wrk_active_times.append((e[0], e[2]))
    ps_idle_times.sort(key=lambda x: x[1])
    wrk_active_times.sort(key=lambda x: x[1])
    rt_job_packs = []
    wrk_active_index = 0
    for job_id, ps_idle in ps_idle_times:
        if wrk_active_index >= len(wrk_active_times):
            rt_job_packs.append(job_id)
            continue
        while wrk_active_index < len(wrk_active_times):
            job_id_wrk = wrk_active_times[wrk_active_index][0]
            wrk_active = wrk_active_times[wrk_active_index][1]
            wrk_active_index += 1
            if wrk_active <= ps_idle:
                rt_job_packs.append((job_id, job_id_wrk))
                break
    return rt_job_packs  # [job_id, (job_id, job_id), ...]


# Also exported for out-package use
# Old design
def wrk_wrk_packing(jobs_data: dict):
    # jobs_data: {job_id: (cpu_active, gpu_active)}
    keys = []
    for key in jobs_data.keys():
        keys.append(key)
    job_pairs = []
    for i in range(len(keys)):
        job1_id = keys[i]
        for j in range(i + 1, len(keys)):
            job2_id = keys[j]
            job_pairs.append([job1_id, job2_id])
    for e in job_pairs:
        j1_id = e[0]
        j2_id = e[1]
        j1_cpu_active = jobs_data[j1_id][0]
        j1_gpu_active = jobs_data[j1_id][1]
        j2_cpu_active = jobs_data[j2_id][0]
        j2_gpu_active = jobs_data[j2_id][1]
        T = np.max([j1_cpu_active, j2_gpu_active]) + np.max([j1_gpu_active, j2_cpu_active])
        gamma = 1 - (T + T - j1_cpu_active - j2_cpu_active - j1_gpu_active - j2_gpu_active) / T / 2
        e.append(gamma)
    result = find_maximum_matching(job_pairs).edge
    keys_set = set(keys)
    for e in result:
        for v in e:
            keys_set.discard(v)
    result.extend(keys_set)
    return result


# Used for wrk-wrk packing
# Old design
gpu_in_use_flags = {}  # "serverid-gpuid: 0"


def _wrk_wrk_packing_connection_thread(connm: znet.SocketMsger):
    key = "server_id-gpu_id"
    while True:
        data = connm.recv()
        if data is None:
            return
        cmd = data[0]
        if cmd == "INIT":
            # job_name = data[1]
            # wrk_id = data[2]
            server_id = data[1]
            gpu_id = data[2]
            key = f"{server_id}-{gpu_id}"
            if f"{server_id}-{gpu_id}" not in gpu_in_use_flags:
                gpu_in_use_flags[f"{server_id}-{gpu_id}"] = 0
        elif cmd == "CHK":
            connm.send(gpu_in_use_flags[key])
        elif cmd == "SET":
            value = int(data[1])
            gpu_in_use_flags[key] = value


def _run_wrk_wrk_packing_server_thread():
    listener = znet.SocketMsger.tcp_listener("0.0.0.0", 62232)
    while True:
        connm, _ = listener.accept()
        threading.Thread(target=_wrk_wrk_packing_connection_thread, args=(connm,)).start()


def run_wrk_wrk_packing_server():
    threading.Thread(target=_run_wrk_wrk_packing_server_thread).start()


if __name__ == "__main__":
    run_wrk_wrk_packing_server()

    # Microsoft philly trace
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    logging.info("Read arriving time of jobs")
    file = open("./philly_trace/philly_time_list.txt", mode="r")
    t0 = time.time()
    time_list = []
    for t in file.readlines():
        hour, minute, second = t.split(":")
        if int(hour) > 17 and int(hour) < 19:
            start_time = 3600 * (int(hour) - 18) + 60 * int(minute) + int(second)
            time_list.append(start_time)

    # Jobs start from here

    # Start cmd clients
    client_sockms = []
    cmd_listener = znet.SocketMsger.tcp_listener("0.0.0.0", listener_port_for_cmd)
    for _ in range(cmd_client_max_num):
        client_sockm, _ = cmd_listener.accept()
        client_sockms.append(client_sockm)

    # First initialize res_monitor and manager
    mo_env = ""  # construct monitor environment string
    for i in range(len(server_ips)):
        mo_env += f"{i}:{server_ips[i]}_{res_monitor_ports[i]},"
    mo_env = mo_env.rstrip(",")
    for i in range(len(server_ips)):
        client_sockms[i].send(f"python3 {workspace}/res_dist_managing.py --role=monitor --srv_id={i} --mo_env={mo_env}")
    client_sockms[0].send(f"python3 {workspace}/res_dist_managing.py --role=manager --srv_id=0 --mo_env={mo_env}")

    # Start jobs from here
    def start_a_job(job_id, model_name, learning_rate, gpu_ids, client_sockm_ids, worker_num):
        ps_num = 1
        job_name = f"job{job_id}"
        rpc_master_port = job_rpc_master_port_init + job_id
        tree_groups = "OFF"  # tree_group = "0,1-2,3-4,5-6,7"
        wrk_server_ids = ""
        for i in range(2, 2 + worker_num):
            wrk_server_ids += f"{client_sockm_ids[i]},"
        wrk_server_ids = wrk_server_ids.rstrip(",")
        for i in range(ps_num + worker_num + 2):
            run_remote_py(
                client_sockms[client_sockm_ids[i]],
                job_name=job_name,
                model_name=model_name,
                rpc_rank=i,
                ps_num=ps_num,
                worker_num=worker_num,
                rpc_master_addr=server_ips[client_sockm_ids[1]],
                rpc_master_port=rpc_master_port,
                epoch_num=10000000000,
                data_partitioned=1,
                gpu_ids=gpu_ids,
                ps_server_id=client_sockm_ids[1],
                wrk_server_ids=wrk_server_ids,
                wrk_pack_toggles="0,0,0,0,0,0,0,0",
                tree_groups=tree_groups,
                lr=learning_rate,
            )

    job_id = 0
    model_name = "googlenet"
    learning_rate = 0.05
    gpu_ids = "0,0,1,0,0"
    client_sockm_ids = [0, 0, 1, 1, 2, 3]
    worker_num = 3
    start_a_job(job_id, model_name, learning_rate, gpu_ids, client_sockm_ids, worker_num)

    job_id = 1
    model_name = "mobilenet"
    learning_rate = 0.05
    gpu_ids = "3,0,1,1,1"
    client_sockm_ids = [1, 1, 0, 0, 2, 3]
    worker_num = 3
    start_a_job(job_id, model_name, learning_rate, gpu_ids, client_sockm_ids, worker_num)

    job_id = 2
    model_name = "alexnet"
    learning_rate = 0.05
    gpu_ids = "0,2,3,2,3"
    client_sockm_ids = [2, 2, 0, 0, 1, 1]
    worker_num = 3
    start_a_job(job_id, model_name, learning_rate, gpu_ids, client_sockm_ids, worker_num)
