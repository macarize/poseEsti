import pickle
import socket
import struct
import torch
import cv2
import os

from test import run_demo, ImageReader
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

HOST = ''
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED

net = PoseEstimationWithMobileNet()
checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)

while True:
    conn, addr = s.accept()
    print('ACCENPTED')

    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)
    print('MESSAGESIZE')

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)
    print('RECIEVED')

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame
    frame = pickle.loads(frame_data)

    pose = run_demo(net, frame, 256, 0, 0, 1)
    if pose is not None:
        print('you')
        pose = pickle.dumps(pose)
        conn.sendall(pose)
        print('SENEDED')

    else:
        print("no keypoints detected")


