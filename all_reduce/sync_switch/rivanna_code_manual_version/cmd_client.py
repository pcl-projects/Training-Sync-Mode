#! /usr/bin/env python3


# from sklearn.cluster import KMeans
# import pickle
# from keras.models import load_model
import random
import socket
import subprocess
from multiprocessing import Process

import numpy as np
import pandas as pd

ClientSocket = socket.socket()
# host = 'udc-aw29-24a.hpc.virginia.edu'
host = "udc-ba26-24.hpc.virginia.edu"
port = 30000


def f(command):
    subprocess.call(command, shell=True)


print("Waiting for connection from s")
try:
    ClientSocket.connect((host, port))
except socket.error as e:
    print(str(e))


# Response = ClientSocket.recv(1024)
while True:
    # Input = input('Say Something: ')
    # ClientSocket.send(str.encode(Input))
    Response = ClientSocket.recv(1024)
    # print(Response.decode('utf-8'))
    command = Response.decode("utf-8")
    p = Process(target=f, args=(command,))
    p.start()

ClientSocket.close()
