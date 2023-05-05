#! /usr/bin/env python3


import threading

import numpy as np
from blossom import find_maximum_matching
from zeyu_utils import net as znet

cmd_listener_port = 30009
cmd_client_max_num = 5
rpc_master_addrs = ["172.31.92.17", "172.31.89.15", "172.31.85.164", "172.31.88.45", "172.31.80.104"]
# rpc_master_addrs = ["3.82.160.143", "52.201.233.211", "3.84.59.239", "18.207.221.103", "44.211.250.101"]


def run_remote_py(sockm, pyfname="training_sgd_allreduce.py", path_prefix="/home/ubuntu/sgd/", **kwargs):
    job_name = kwargs["job_name"]
    model_name = kwargs["model_name"]
    rpc_rank = kwargs["rpc_rank"]
    # ps_num = kwargs["ps_num"]
    worker_num = kwargs["worker_num"]
    rpc_master_addr = kwargs["rpc_master_addr"]
    rpc_master_port = kwargs["rpc_master_port"]
    epoch_num = kwargs["epoch_num"]
    wrk_server_ids = kwargs["wrk_server_ids"]
    wrk_pack_toggles = kwargs["wrk_pack_toggles"]
    data_partitioned = kwargs["data_partitioned"]
    gpu_ids = kwargs["gpu_ids"]
    policy = kwargs["policy"]

    lr = kwargs["lr"]
    # model_fetch_ids = kwargs["model_fetch_ids"]

    # gpu_id_str = ""
    # for gpu_id in gpu_ids:
    #     gpu_id_str += f"{gpu_id} "
    # gpu_id_str = gpu_id_str.rstrip()

    cmd = f"python3 {path_prefix}/{pyfname} --job_name={job_name} --model_name={model_name} --rpc_rank={rpc_rank} --worker_num={worker_num} --rpc_master_addr={rpc_master_addr} --rpc_master_port={rpc_master_port} --epoch_num={epoch_num} --data_partitioned={data_partitioned} --gpu_ids={gpu_ids} --learning_rate={lr} --wrk_server_ids={wrk_server_ids} --wrk_pack_toggles={wrk_pack_toggles} --init_sync_policy={policy}"

    sockm.send(cmd)


# also exported for out-package use
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


# also exported for out-package use
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


gpu_in_use_flags = {}  # "serverid-gpuid: 0"


def _connection_thread(connm: znet.SocketMsger):
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
        threading.Thread(target=_connection_thread, args=(connm,)).start()


def run_wrk_wrk_packing_server():
    threading.Thread(target=_run_wrk_wrk_packing_server_thread).start()


if __name__ == "__main__":
    run_wrk_wrk_packing_server()

    job_name = "job0"
    model_name = "resnet20"
    ps_num = 1
    worker_num = 4
    # gpu_ids = "0,1,2,3,0,1,2,3,0"
    gpu_ids = "0,0,1,0,1,1"
    # model_fetch_ids = "0,1,2,3"
    # client_sockm_ids = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0]
    client_sockm_ids = [0, 0, 1, 1, 2, 2, 0]

    client_sockms = []
    cmd_listener = znet.SocketMsger.tcp_listener("0.0.0.0", cmd_listener_port)
    for _ in range(cmd_client_max_num):
        client_sockm, _ = cmd_listener.accept()
        client_sockms.append(client_sockm)

    master_port = 49670
    ps_num = 0
    worker_num = 8
    tree_group = "0,1-2,3-4,5-6,7"
    tree_group = "OFF"

    job_name = "job0"
    model_name = "resnet20"
    model_name = "densenet121"
    master_port += 1
    client_sockm_ids = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]
    wrk_server_ids = "1,1,2,2,3,3,4,4"
    gpu_ids = "0,1,0,1,0,1,0,1,0"
    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            worker_num=worker_num,
            rpc_master_addr=rpc_master_addrs[0],
            rpc_master_port=master_port,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            wrk_server_ids=wrk_server_ids,
            wrk_pack_toggles="0,0,0,0,0,0,0,0",
            lr=0.01,
            policy="None",
            # model_fetch_ids=model_fetch_ids,
        )

    # job_name = "job1"
    # model_name = "resnet56"
    # model_name = "googlenet"
    # master_port += 1
    # client_sockm_ids = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]
    # wrk_server_ids = "1,1,2,2,3,3,4,4"
    # gpu_ids = "0,1,0,1,0,1,0,1,1"
    # for i in range(ps_num + worker_num + 2):
    #     run_remote_py(
    #         client_sockms[client_sockm_ids[i]],
    #         job_name=job_name,
    #         model_name=model_name,
    #         rpc_rank=i,
    #         worker_num=worker_num,
    #         rpc_master_addr=rpc_master_addrs[0],
    #         rpc_master_port=master_port,
    #         epoch_num=10000000000,
    #         data_partitioned=1,
    #         gpu_ids=gpu_ids,
    #         wrk_server_ids=wrk_server_ids,
    #         wrk_pack_toggles="1,1,1,1,1,1,1,1",
    #         lr=0.01,
    #         policy="None",
    #         # model_fetch_ids=model_fetch_ids,
    #     )

    job_name = "job2"
    model_name = "alexnet"
    model_name = "googlenet"
    master_port += 1
    client_sockm_ids = [1, 2, 2, 3, 3, 4, 4, 0, 0, 1]
    wrk_server_ids = "2,2,3,3,4,4,0,0"
    gpu_ids = "2,3,2,3,2,3,2,3,2"
    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            worker_num=worker_num,
            rpc_master_addr=rpc_master_addrs[1],
            rpc_master_port=master_port,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            wrk_server_ids=wrk_server_ids,
            wrk_pack_toggles="0,0,0,0,0,0,0,0",
            lr=0.01,
            policy="0,2,4,6S1C0,3C2,5C4,7C6",
            # model_fetch_ids=model_fetch_ids,
        )

    # job_name = "job3"
    # model_name = "mobilenet"
    # model_name = "googlenet"
    # master_port += 1
    # client_sockm_ids = [1, 2, 2, 3, 3, 4, 4, 0, 0, 1]
    # wrk_server_ids = "2,2,3,3,4,4,0,0"
    # gpu_ids = "2,3,2,3,2,3,2,3,3"
    # for i in range(ps_num + worker_num + 2):
    #     run_remote_py(
    #         client_sockms[client_sockm_ids[i]],
    #         job_name=job_name,
    #         model_name=model_name,
    #         rpc_rank=i,
    #         worker_num=worker_num,
    #         rpc_master_addr=rpc_master_addrs[1],
    #         rpc_master_port=master_port,
    #         epoch_num=10000000000,
    #         data_partitioned=1,
    #         gpu_ids=gpu_ids,
    #         wrk_server_ids=wrk_server_ids,
    #         wrk_pack_toggles="1,1,1,1,1,1,1,1",
    #         lr=0.01,
    #         policy="0,2,4,6S1C0,3C2,5C4,7C6",
    #         # model_fetch_ids=model_fetch_ids,
    #     )

    job_name = "job4"
    model_name = "vgg13"
    model_name = "googlenet"
    master_port += 1
    client_sockm_ids = [2, 3, 3, 4, 4, 0, 0, 1, 1, 2]
    wrk_server_ids = "3,3,4,4,0,0,1,1"
    gpu_ids = "4,5,4,5,4,5,4,5,4"
    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            worker_num=worker_num,
            rpc_master_addr=rpc_master_addrs[2],
            rpc_master_port=master_port,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            wrk_server_ids=wrk_server_ids,
            wrk_pack_toggles="0,0,0,0,0,0,0,0",
            lr=0.01,
            policy="0S1C0,2C0,3C0,4C0,5C0,6C0,7C0",
            # model_fetch_ids=model_fetch_ids,
        )

    # job_name = "job5"
    # model_name = "vgg16"
    # model_name = "googlenet"
    # master_port += 1
    # client_sockm_ids = [2, 3, 3, 4, 4, 0, 0, 1, 1, 2]
    # wrk_server_ids = "3,3,4,4,0,0,1,1"
    # gpu_ids = "4,5,4,5,4,5,4,5,5"
    # for i in range(ps_num + worker_num + 2):
    #     run_remote_py(
    #         client_sockms[client_sockm_ids[i]],
    #         job_name=job_name,
    #         model_name=model_name,
    #         rpc_rank=i,
    #         worker_num=worker_num,
    #         rpc_master_addr=rpc_master_addrs[2],
    #         rpc_master_port=master_port,
    #         epoch_num=10000000000,
    #         data_partitioned=1,
    #         gpu_ids=gpu_ids,
    #         wrk_server_ids=wrk_server_ids,
    #         wrk_pack_toggles="1,1,1,1,1,1,1,1",
    #         lr=0.01,
    #         policy="0S1C0,2C0,3C0,4C0,5C0,6C0,7C0",
    #         # model_fetch_ids=model_fetch_ids,
    #     )

    job_name = "job6"
    model_name = "lstm"
    model_name = "densenet121"
    master_port += 1
    client_sockm_ids = [3, 4, 4, 0, 0, 1, 1, 2, 2, 3]
    wrk_server_ids = "4,4,0,0,1,1,2,2"
    gpu_ids = "6,7,6,7,6,7,6,7,6"
    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            worker_num=worker_num,
            rpc_master_addr=rpc_master_addrs[3],
            rpc_master_port=master_port,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            wrk_server_ids=wrk_server_ids,
            wrk_pack_toggles="0,0,0,0,0,0,0,0",
            lr=0.01,
            policy="None",
            # policy="0,1,2,3S4C0,5C1,6C2,7C3",
            # model_fetch_ids=model_fetch_ids,
        )

    # job_name = "job7"
    # model_name = "transformer"
    # model_name = "googlenet"
    # master_port += 1
    # client_sockm_ids = [3, 4, 4, 0, 0, 1, 1, 2, 2, 3]
    # wrk_server_ids = "4,4,0,0,1,1,2,2"
    # gpu_ids = "6,7,6,7,6,7,6,7,7"
    # for i in range(ps_num + worker_num + 2):
    #     run_remote_py(
    #         client_sockms[client_sockm_ids[i]],
    #         job_name=job_name,
    #         model_name=model_name,
    #         rpc_rank=i,
    #         worker_num=worker_num,
    #         rpc_master_addr=rpc_master_addrs[3],
    #         rpc_master_port=master_port,
    #         epoch_num=10000000000,
    #         data_partitioned=1,
    #         gpu_ids=gpu_ids,
    #         wrk_server_ids=wrk_server_ids,
    #         wrk_pack_toggles="1,1,1,1,1,1,1,1",
    #         lr=0.01,
    #         policy="0,1,2,3S4C0,5C1,6C2,7C3",
    #     )
