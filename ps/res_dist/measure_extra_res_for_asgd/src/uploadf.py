#! /usr/bin/env python3

from zeyu_utils import os as zos

ips = ["3.95.208.124", "184.72.193.184"]

for ip in ips:
    cmd = f"scp -r ubuntu@{ip}:/home/ubuntu/sgd/training_logs/* ./res_dist/measure_extra_res_for_asgd/logs/"
    # cmd = f"scp -r switch_gen_strg/src/models ubuntu@{ip}:/home/ubuntu/sgd/"
    # cmd = f"scp -r ./res_dist/measure_extra_res_for_asgd/src/*.py ubuntu@{ip}:/home/ubuntu/sgd/"
    zos.run_cmd(cmd)
