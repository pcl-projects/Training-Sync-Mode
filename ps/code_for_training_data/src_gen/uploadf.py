#! /usr/bin/env python3

from zeyu_utils import os as zos

DIR = "/Users/zeyu/Documents/Seafile/CS Learning/Paper Projects/Sync & Async Para Updating/Experiment/Paper Exp./code_for_training_data/src/"

ips = ["3.92.227.80", "44.203.3.186", "44.204.47.32", "3.95.164.32", "44.202.133.134"]

for ip in ips:
    cmd = f"scp -r ubuntu@{ip}:/home/ubuntu/sgd/training_logs/* code_for_training_data/logs/"
    # cmd = f"scp -r code_for_training_data/src/*.py ubuntu@{ip}:/home/ubuntu/sgd/"
    zos.run_cmd(cmd)
