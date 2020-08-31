import cv2
import numpy as np
import socket
import sys
import pickle
import struct

cap=cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('54.180.176.99',8089))

while True:
    ret,frame=cap.read()
    # Serialize frame
    data = pickle.dumps(frame)
    print('1')
    # Send message length first
    message_size = struct.pack("L", len(data))
    print('2')

    # Then data
    clientsocket.sendall(message_size + data)
    print('3')

    pose = clientsocket.recv(4096)
    pose = pickle.loads(pose)
    print(pose.keypoints)
    print('4')

    pose.draw(frame)
    cv2.imshow('test', frame)
    if cv2.waitKey(1) == ord('q'):
       break
