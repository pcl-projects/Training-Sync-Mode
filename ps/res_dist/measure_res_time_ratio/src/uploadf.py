#! /usr/bin/env python3

from zeyu_utils import os as zos

ips = ["44.204.161.33", "3.83.242.18"]

for ip in ips:
    cmd = f"scp -r ubuntu@{ip}:/home/ubuntu/sgd/training_logs/* ./res_dist/measure_res_time_ratio/logs/"
    # cmd = f"scp -r switch_gen_strg/src/models ubuntu@{ip}:/home/ubuntu/sgd/"
    # cmd = f"scp -r ./res_dist/measure_res_time_ratio/src/*.py ubuntu@{ip}:/home/ubuntu/sgd/"
    zos.run_cmd(cmd)
