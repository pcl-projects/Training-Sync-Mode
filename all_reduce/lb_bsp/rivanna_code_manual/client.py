#! /usr/bin/env python3


import time

from zeyu_utils import net as znet

connm = znet.SocketMsger.tcp_connect("127.0.0.1", 14566)

connm.send("3")
connm.close()
