import argparse
import multiprocessing as mp
import threading as td
import time

import psutil
from zeyu_utils import net as znet

BW_NIC = "ib0"
RECORD_PERIOD = 1

SRV_CPU_CAPA = 100
SRV_BW_CAPA = 1221.12  # MB/s  # 9.54 Gbits/sec
SRV_BW_CAPA = 100  # MB/s  # 9.54 Gbits/sec

MODEL_RES_TIME_RATIO = {
    "resnet20": {"cpu": 0.91, "bw": 1.285},
    "resnet56": {"cpu": 0.856, "bw": 1.542},
    "alexnet": {"cpu": 1.156, "bw": 25.87},
    "mobilenet": {"cpu": 1.048, "bw": 3.16},
    "googlenet": {"cpu": 0.644, "bw": 0.027},
    "densenet121": {"cpu": 0.838, "bw": 0.91},
    "vgg13": {"cpu": 1.206, "bw": 34.355},
    "vgg16": {"cpu": 1.288, "bw": 42.515},
    "lstm": {"cpu": 1.244, "bw": 62.277},
    "transformer": {"cpu": 1.48, "bw": 62.226},
}
MODEL_RES_SENSITIVITY = {
    "resnet20": {"cpu": 1760 * 1000 / 600 / 600, "bw": 600 * 560 / 600 / 600},
    "resnet56": {"cpu": 2400 * 1320 / 920 / 920, "bw": 1040 * 800 / 920 / 920},
    "alexnet": {"cpu": 2400 * 1240 / 680 / 680, "bw": 2400 * 1360 / 680 / 680},
    "mobilenet": {"cpu": 1560 * 920 / 480 / 480, "bw": 1080 * 560 / 480 / 480},
    "googlenet": {"cpu": 1200 * 720 / 480 / 480, "bw": 1040 * 560 / 480 / 480},
    "densenet121": {"cpu": 1400 * 860 / 560 / 560, "bw": 1320 * 920 / 560 / 560},
    "vgg13": {"cpu": 480 * 280 / 160 / 160, "bw": 1280 * 560 / 160 / 160},
    "vgg16": {"cpu": 1120 * 640 / 560 / 560, "bw": 4320 * 760 / 560 / 560},
    "lstm": {"cpu": 720 * 380 / 260 / 260, "bw": 2160 * 1060 / 260 / 260},
    "transformer": {"cpu": 800 * 530 / 310 / 310, "bw": 2620 * 1210 / 310 / 310},
}
MODEL_EXTRA_RES_FOR_ASGD = {
    "resnet20": {"cpu": 18.86, "bw": 3.675},
    "resnet56": {"cpu": 11.8, "bw": 0.839},
    "alexnet": {"cpu": 3.41, "bw": 22.614},
    "mobilenet": {"cpu": 4.71, "bw": 3.402},
    "googlenet": {"cpu": 3.84, "bw": 6.491},
    "densenet121": {"cpu": 4.64, "bw": 12},
    "vgg13": {"cpu": 40, "bw": 178.768},
    "vgg16": {"cpu": 41.26, "bw": 218.587},
    "lstm": {"cpu": 36.74, "bw": 270.048},
    "transformer": {"cpu": 38.92, "bw": 258.758},
}


class ResMonitor:
    def __init__(self, srv_id, env) -> None:
        self.srv_id = srv_id
        self.env = env
        self.proc = None
        # define some shared structures for threads
        self.res = {"cpu": [0.0], "in_bw": [0.0], "out_bw": [0.0]}

    def start(self):
        self.proc = mp.Process(target=self.proc_func_for_start)
        self.proc.start()

    def join(self):
        self.proc.join()

    def proc_func_for_start(self):
        port = self.env[self.srv_id][1]
        listener = znet.SocketMsger.tcp_listener("0.0.0.0", port)

        # run monitor thread
        td.Thread(target=self.start_monitor_thread).start()

        while True:
            sockm, _ = listener.accept()
            td.Thread(target=self.start_sockm_thread, args=(sockm,)).start()

    def start_monitor_thread(self):
        cpu_slot = self.res["cpu"]
        in_bw_slot = self.res["in_bw"]
        out_bw_slot = self.res["out_bw"]
        psutil.cpu_percent()
        inbw0 = psutil.net_io_counters(pernic=True)[BW_NIC].bytes_recv
        outbw0 = psutil.net_io_counters(pernic=True)[BW_NIC].bytes_sent
        time0 = time.time()
        while True:
            time.sleep(RECORD_PERIOD)
            cpu_slot[0] = psutil.cpu_percent()
            inbw1 = psutil.net_io_counters(pernic=True)[BW_NIC].bytes_recv
            outbw1 = psutil.net_io_counters(pernic=True)[BW_NIC].bytes_sent
            time1 = time.time()
            time_diff = time1 - time0
            in_bw_slot[0] = (inbw1 - inbw0) / 1048576 / time_diff
            out_bw_slot[0] = (outbw1 - outbw0) / 1048576 / time_diff
            inbw0 = inbw1
            outbw0 = outbw1
            time0 = time1
            # print(cpu_slot, in_bw_slot, out_bw_slot)

    def start_sockm_thread(self, sockm: znet.SocketMsger):
        msg = sockm.recv()
        while msg is not None:
            cmd, data = msg
            if cmd == "GET":
                if data == "RES":
                    sockm.send((self.res["cpu"][0], self.res["in_bw"][0], self.res["out_bw"][0]))
            msg = sockm.recv()
        sockm.close()


class Manager:
    def __init__(self, res_mo_env, mp_sync_mngr) -> None:
        self.rmoenv = res_mo_env
        self.mo_sockm = {}
        self.job_srv_info = mp_sync_mngr.dict()
        self.srv_job_info = mp_sync_mngr.dict()
        self.job_iter_time = mp_sync_mngr.dict()
        self.job_rsockm = mp_sync_mngr.dict()
        self.job_mdl_name = mp_sync_mngr.dict()
        self.job_step_num = mp_sync_mngr.dict()

        self.job_mode = mp_sync_mngr.dict()

        for k, _ in res_mo_env.items():
            self.mo_sockm[k] = None
            self.srv_job_info[k] = mp_sync_mngr.dict()

    def start(self):
        self.proc = mp.Process(target=self.proc_func_for_start)
        self.proc.start()

    def join(self):
        self.proc.join()

    def proc_func_for_start(self):
        port = 48848
        listener = znet.SocketMsger.tcp_listener("0.0.0.0", port)

        for k, itm in self.rmoenv.items():
            sockm = znet.SocketMsger.tcp_connect(itm[0], itm[1])
            self.mo_sockm[k] = sockm

        while True:
            sockm, _ = listener.accept()
            mp.Process(target=self.start_job_sockm_process, args=(sockm,)).start()

    def start_job_sockm_process(self, sockm: znet.SocketMsger):
        jname, mname = sockm.recv()
        # self.job_sockm[jname] = sockm
        msg = sockm.recv()
        while msg is not None:
            cmd, data = msg
            if cmd == "IF_ASGD":
                ssgd_delay = data
                extra_res = MODEL_EXTRA_RES_FOR_ASGD[mname]
                ps_srv_id = self.job_srv_info[jname]["ps"]
                td.Thread(target=self._determine_if_asgd, args=(sockm, jname, ps_srv_id, extra_res, ssgd_delay)).start()
                # if_asgd = self._determine_if_asgd(jname, ps_srv_id, extra_res, ssgd_delay)
                # if if_asgd:
                #     sockm.send("ASGD")
                # else:
                #     sockm.send("SSGD")
            elif cmd == "START_JOB":
                srv_info = data
                self.job_srv_info[jname] = srv_info
                self.job_mdl_name[jname] = mname
                for task, srv_id in srv_info.items():
                    self.srv_job_info[srv_id][f"{jname}_{task}"] = None
                self.job_mode[jname] = 0
                self.job_step_num[jname] = 0
            elif cmd == "END_JOB":
                # jname = data

                jobs = {}
                ps_srv_id = self.job_srv_info[jname]["ps"]
                for task in self.srv_job_info[ps_srv_id].keys():
                    jn, tsk = task.split("_")
                    if "wrk" in tsk:
                        jobs[jn] = None
                jobs.pop(jname, None)
                for key in jobs.keys():
                    rsockm = self.job_rsockm[key]
                    rsockm.send(("UNSET_DLY", None))

                srv_info = self.job_srv_info[jname]
                for task, srv_id in srv_info.items():
                    self.srv_job_info[srv_id].pop(f"{jname}_{task}", None)
                self.job_srv_info.pop(jname, None)
                self.job_iter_time.pop(jname, None)
                self.job_mode.pop(jname, None)
                self.job_step_num.pop(jname, None)
                rsockm = self.job_rsockm.pop(jname, None)
                if rsockm is not None:
                    rsockm.close()

                break
            elif cmd == "SSGD":
                self.job_mode[jname] = 0
                jobs = {}
                ps_srv_id = self.job_srv_info[jname]["ps"]
                for task in self.srv_job_info[ps_srv_id].keys():
                    jn, tsk = task.split("_")
                    if "wrk" in tsk:
                        jobs[jn] = None
                jobs.pop(jname, None)
                for key in jobs.keys():
                    rsockm = self.job_rsockm[key]
                    rsockm.send(("UNSET_DLY", None))
            elif cmd == "ITER_TIME":
                iter_t_info = data
                self.job_iter_time[jname] = iter_t_info
            elif cmd == "EST_RCONN":
                self.job_rsockm[jname] = sockm
                return
            elif cmd == "STEP_NUM":
                self.job_step_num[jname] = data
            msg = sockm.recv()
        sockm.close()

    def _determine_if_asgd(self, sockm, job_name, ps_srv_id, extra_res, ssgd_delay):
        self.mo_sockm[ps_srv_id].send(("GET", "RES"))
        cpu, in_bw, out_bw = self.mo_sockm[ps_srv_id].recv()
        ava_cpu = SRV_CPU_CAPA - cpu
        ava_in_bw = SRV_BW_CAPA - in_bw
        ava_out_bw = SRV_BW_CAPA - out_bw
        extra_cpu = extra_res["cpu"]
        extra_bw = extra_res["bw"]

        jobs = {}  # job_name: [wrk_id, ...]

        # compare resource
        if extra_cpu > ava_cpu or extra_bw > ava_in_bw or extra_bw > ava_out_bw:
            for task in self.srv_job_info[ps_srv_id].keys():
                jname, tsk = task.split("_")
                if self.job_mode[jname] == 1:
                    continue
                if "wrk" in tsk:
                    if jname in jobs:
                        jobs[jname].append(int(tsk[3:]))
                    else:
                        jobs[jname] = [int(tsk[3:])]
            jobs.pop(job_name, None)
            for key in jobs.keys():
                mdl_name = self.job_mdl_name[key]
                if key in self.job_iter_time:
                    max_iter_t = float("-inf")
                    for _, itm in self.job_iter_time[key].items():
                        if itm > max_iter_t:
                            max_iter_t = itm
                    for wid in jobs[key]:
                        delayed_t = max_iter_t - self.job_iter_time[key][wid]
                        ava_cpu += delayed_t / 1000 * MODEL_RES_TIME_RATIO[mdl_name]["cpu"]
                        more_bw = delayed_t / 1000 * MODEL_RES_TIME_RATIO[mdl_name]["bw"]
                        ava_in_bw += more_bw
                        ava_out_bw += more_bw

        if extra_cpu <= ava_cpu and extra_bw <= ava_in_bw and extra_bw <= ava_out_bw:
            for key in jobs.keys():
                rsockm = self.job_rsockm[key]
                rsockm.send(("DELAY_WRK", 0))
            sockm.send("ASGD")
            self.job_mode[job_name] = 1

        # continue to check if res are enough
        if len(jobs) == 0:
            for task in self.srv_job_info[ps_srv_id].keys():
                jname, tsk = task.split("_")
                if self.job_mode[jname] == 1:
                    continue
                if "wrk" in tsk:
                    if jname in jobs:
                        jobs[jname].append(int(tsk[3:]))
                    else:
                        jobs[jname] = [int(tsk[3:])]
        jobs.pop(job_name, None)

        de_sens_cpu_sum = 0.0
        de_sens_bw_sum = 0.0
        for key in jobs.keys():
            mdl_name = self.job_mdl_name[key]
            de_sens_cpu_sum += 1 / MODEL_RES_SENSITIVITY[mdl_name]["cpu"]
            de_sens_bw_sum += 1 / MODEL_RES_SENSITIVITY[mdl_name]["bw"]

        job_delay_t = {}

        cpu_needed = extra_cpu - ava_cpu
        bw_needed = extra_bw - min(ava_in_bw, ava_out_bw)
        for key in jobs.keys():
            mdl_name = self.job_mdl_name[key]
            v = 0
            if cpu_needed > 0:
                v = (
                    (1 / MODEL_RES_SENSITIVITY[mdl_name]["cpu"] / de_sens_cpu_sum)
                    * cpu_needed
                    / len(jobs[key])
                    / MODEL_RES_TIME_RATIO[mdl_name]["cpu"]
                ) * (1 + self.job_step_num[key] / 10000)
            if bw_needed > 0:
                temp = (
                    (1 / MODEL_RES_SENSITIVITY[mdl_name]["bw"] / de_sens_bw_sum)
                    * bw_needed
                    / len(jobs[key])
                    / MODEL_RES_TIME_RATIO[mdl_name]["bw"]
                ) * (1 + self.job_step_num[key] / 10000)
                if temp > v:
                    v = temp
            job_delay_t[key] = v

        delay_sum = 0
        for _, itm in job_delay_t.items():
            delay_sum += itm
        if ssgd_delay <= delay_sum:
            sockm.send("SSGD")
            self.job_mode[job_name] = 0
        else:
            for key in jobs.keys():
                rsockm = self.job_rsockm[key]
                rsockm.send(("DELAY_WRK", job_delay_t[key]))
            sockm.send("ASGD")
            self.job_mode[job_name] = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, default="monitor")
    parser.add_argument("--srv_id", type=int, default=0)
    parser.add_argument("--mo_env", type=str, default="0:127.0.0.1_65535,1:127.0.0.1_65535")

    args = parser.parse_args()

    env = args.mo_env.split(",")
    mo_env = {}
    for i in range(len(env)):
        string = env[i]
        array = string.split(":")
        srv_id = int(array[0])
        ip = array[1].split("_")[0]
        port = int(array[1].split("_")[1])
        mo_env[srv_id] = (ip, port)

    if args.role == "monitor":
        monitor = ResMonitor(args.srv_id, mo_env)
        monitor.start()
        monitor.join()
    elif args.role == "manager":
        with mp.Manager() as manager:
            res_mngr = Manager(mo_env, manager)
            res_mngr.start()
            res_mngr.join()
