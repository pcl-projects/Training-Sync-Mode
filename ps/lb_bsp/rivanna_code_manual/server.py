#! /usr/bin/env python3


import time

from zeyu_utils import net as znet

listener = znet.SocketMsger.tcp_listener("127.0.0.1", 14566)

connm, _ = listener.accept()

connm.recv()

while True:
    time.sleep(3)
    print(connm.closed)
    connm.send("3")
