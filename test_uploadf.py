#! /usr/bin/env python3

from zeyu_utils import os as zos

ips = ["174.129.155.205", "18.207.220.225", "3.83.134.190", "3.93.81.227", "100.26.201.246"]

for ip in ips:
    cmd = f"scp -r ubuntu@{ip}:/home/ubuntu/sgd/training_logs/* ./logs/"
    # cmd = f"scp -r all_reduce/*.py ubuntu@{ip}:/home/ubuntu/sgd/"
    zos.run_cmd(cmd)
