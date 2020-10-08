import cv2
import numpy as np
import socket
import sys
import pickle
import struct
from flask import Flask, render_template, Response
import io

stepA = False
stepB = False
count = 0

sitAngle = 0
stdupAngle = 0

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50, 400)
topLeft = (150, 400)

fontScale = 3
fontColor = (255, 0, 0)
lineType = 2


cap=cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('54.180.176.99',8089))
#cap = cv2.VideoCapture('pushUp.mp4')
while True:
    ret,frame=cap.read()
    # Serialize frame
    #frame = cv2.transpose(frame)
    #frame = cv2.flip(frame,flipCode=1)
    data = pickle.dumps(frame)
    # Send message length first
    message_size = struct.pack("=L", len(data))

    # Then data
    clientsocket.sendall(message_size + data)

