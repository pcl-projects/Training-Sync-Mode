#! /usr/bin/env python3


from datetime import datetime

import numpy as np

LOG_DIR = "/Users/zeyu/Documents/Seafile/CS Learning/Paper Projects/Sync & Async Para Updating/Experiment/Paper Exp./res_dist/measure_extra_res_for_asgd/logs"


def get_res(mdl_name, res_type="cpu"):
    switch_time = 0
    with open(f"{LOG_DIR}/{mdl_name}_ps0.log") as f:
        line = f.readline()
        while line != "":
            if "to_asgd" in line:
                arr = line.rstrip("\n").split()
                t = datetime.timestamp(datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f"))
                switch_time = t
                break
            line = f.readline()

    if res_type == "cpu":
        cpu_pre = []
        cpu_post = []
        with open(f"{LOG_DIR}/{mdl_name}_cpu.log") as f:
            line = f.readline()
            while line != "":
                arr = line.rstrip("\n").split()
                t = datetime.timestamp(datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f"))
                if t >= switch_time - 10 and t <= switch_time:
                    cpu_pre.append(float(arr[-1]))
                elif t > switch_time and t <= switch_time + 10:
                    cpu_post.append(float(arr[-1]))
                line = f.readline()
    elif res_type == "bw":
        in_bw_pre = []
        in_bw_post = []
        out_bw_pre = []
        out_bw_post = []
        with open(f"{LOG_DIR}/{mdl_name}_bw.log") as f:
            line = f.readline()
            while line != "":
                arr = line.rstrip("\n").split()
                t = datetime.timestamp(datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f"))
                if t >= switch_time - 10 and t <= switch_time:
                    in_bw_pre.append(float(arr[-2]))
                    out_bw_pre.append(float(arr[-1]))
                elif t > switch_time and t <= switch_time + 10:
                    in_bw_post.append(float(arr[-2]))
                    out_bw_post.append(float(arr[-1]))
                line = f.readline()
    if res_type == "cpu":
        return np.average(cpu_post) - np.average(cpu_pre)
    elif res_type == "bw":
        in_bw_avg_diff = np.average(in_bw_post) - np.average(in_bw_pre)
        out_bw_avg_diff = np.average(out_bw_post) - np.average(out_bw_pre)
        return np.average([in_bw_avg_diff, out_bw_avg_diff])


print(get_res("transformer", "bw"))
