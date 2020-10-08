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

    pose = clientsocket.recv(4096)
    pose = pickle.loads(pose)

    pose.draw(frame)
    cv2.imshow('test', frame)
    if cv2.waitKey(1) == ord('q'):
       break

    A = np.array([pose.keypoints[8][0],pose.keypoints[8][1]])
    B = np.array([pose.keypoints[9][0],pose.keypoints[9][1]])
    C = np.array([pose.keypoints[10][0],pose.keypoints[10][1]])

    BA = A - B
    BC = C - B

    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    print(angle)

    if angle > 140 :
        stepA = True
        if stdupAngle is 0:
            stdupAngle = angle
        if angle > stdupAngle:
            stdupAngle = angle
    if angle < 70 :
        stepB = True
        if sitAngle is 0:
            sitAngle = angle
        if angle < sitAngle:
            sitAngle = angle

    if stepA and stepB is True :
        if angle > 140:
            stepA = False
            stepB = False
            count += 1

            if sitAngle > 60:
                cv2.putText(frame,"Bend your knee more",
                            topLeft,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            stdupDiff = 140 - stdupAngle
            sitDiff = 70 - sitAngle

            stdupDiff = abs(stdupDiff)
            sitDiff = abs(sitDiff)
            correctness = 280 - (stdupDiff + sitDiff)

            cv2.putText(frame, "correctness" + str(correctness),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            sitAngle = 0
            stdupAngle = 0
    if stepA is True and stepB is False :
        if angle > 140:
            stepA = False
            stepB = False
            sitAngle = 0
            stdupAngle = 0

    cv2.putText(frame, "count" + str(count),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow("img", frame)

    print(count)