#! /usr/bin/env python3


import subprocess
from multiprocessing import Process

from zeyu_utils import net as znet

host = "172.31.92.17"
port = 30009


def run_cmd(cmd):
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    print("Connecting...")
    client_sockm = znet.SocketMsger.tcp_connect(host, port)

    while True:
        cmd = client_sockm.recv()
        if cmd is None:
            print("CMD client ends.")
            break
        Process(target=run_cmd, args=(cmd,)).start()
