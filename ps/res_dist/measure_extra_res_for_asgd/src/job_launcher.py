#! /usr/bin/env python3

import time

from zeyu_utils import net as znet

cmd_listener_port = 30009
cmd_client_max_num = 2
rpc_master_addr = "gpusrv08.cs.Virginia.EDU"
rpc_master_addr = "172.31.92.17"
# rpc_master_addr = "udc-ba26-26-ic.hpc.virginia.edu"


# def run_remote_py(sockm, pyfname="training_sgd.py", path_prefix="/u/qxc4fh/zeyu_workspace/sgd/", **kwargs):
def run_remote_py(sockm, pyfname="training_sgd.py", path_prefix="/home/ubuntu/sgd/", **kwargs):
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
    # model_fetch_ids = kwargs["model_fetch_ids"]

    # gpu_id_str = ""
    # for gpu_id in gpu_ids:
    #     gpu_id_str += f"{gpu_id} "
    # gpu_id_str = gpu_id_str.rstrip()

    cmd = f"python3 {path_prefix}/{pyfname} --job_name={job_name} --model_name={model_name} --rpc_rank={rpc_rank} --ps_num={ps_num} --worker_num={worker_num} --rpc_master_addr={rpc_master_addr} --rpc_master_port={rpc_master_port} --epoch_num={epoch_num} --data_partitioned={data_partitioned} --gpu_ids={gpu_ids}"

    sockm.send(cmd)


if __name__ == "__main__":
    client_sockms = []
    cmd_listener = znet.SocketMsger.tcp_listener("0.0.0.0", cmd_listener_port)
    for _ in range(cmd_client_max_num):
        client_sockm, _ = cmd_listener.accept()
        client_sockms.append(client_sockm)

    job_name = "job0"
    model_name = "transformer"
    ps_num = 1
    worker_num = 8
    gpu_ids = "0,1,2,3,4,5,6,7,0"
    client_sockm_ids = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            ps_num=ps_num,
            worker_num=worker_num,
            rpc_master_addr=rpc_master_addr,
            rpc_master_port=59670,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            # model_fetch_ids=model_fetch_ids,
        )
    # client_sockms[client_sockm_ids[1]].send(
    #     f"python3 /home/qxc4fh/zeyu_workspace/sgd//job_recorder.py --job_name={job_name} --model_name={model_name} --rpc_rank=1 --res_type=cpu"
    # )

    # job_name = "job1"
    # model_name = "mobilenet"
    # ps_num = 1
    # worker_num = 6
    # gpu_ids = "1,2,3,0,1,1,0"
    # client_sockm_ids = [0, 1, 0, 0, 0, 1, 2, 3, 1]

    # for i in range(ps_num + worker_num + 2):
    #     run_remote_py(
    #         client_sockms[client_sockm_ids[i]],
    #         job_name=job_name,
    #         model_name=model_name,
    #         rpc_rank=i,
    #         ps_num=ps_num,
    #         worker_num=worker_num,
    #         rpc_master_addr=rpc_master_addr,
    #         rpc_master_port=59671,
    #         epoch_num=10000000000,
    #         data_partitioned=1,
    #         gpu_ids=gpu_ids,
    #         do_switch=0,
    #         # model_fetch_ids=model_fetch_ids,
    #     )

    time.sleep(5)

    server_name = "server0"
    task_names = "job0_mobilenet_ps,job0_mobilenet_wrk0,job1_mobilenet_wrk0,job1_mobilenet_wrk1,job1_mobilenet_wrk2"
    client_sockms[0].send(f"python3 /home/ubuntu/sgd/res_recorder.py --server_name={server_name} --model_name={model_name}")
