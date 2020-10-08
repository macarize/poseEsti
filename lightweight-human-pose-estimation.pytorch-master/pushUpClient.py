import cv2
import numpy as np
import socket
import sys
import pickle
import struct


print('pose')
A = 0
B = 0
C = 0

stepA = False
stepB = False
count = 0

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
    ret, frame=cap.read()
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



    A = np.array([pose.keypoints[2][0],pose.keypoints[2][1]])
    if B is 0:
        B = np.array([pose.keypoints[10][0],pose.keypoints[10][1]])
    if C is 0:
        C = np.array([pose.keypoints[4][0],pose.keypoints[4][1]])

    BA = A - B
    BC = C - B

    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    print(angle)

    if angle > 25 :
        stepA = True
    if angle < 15 :
        stepB = True

    if stepA and stepB is True :
        if angle > 25:
            stepA = False
            stepB = False
            count += 1

            cv2.putText(frame,"good",
                        topLeft,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
    if stepA is True and stepB is False :
        if angle > 25:
            stepA = False
            stepB = False
    cv2.putText(frame, "count" + str(count),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow("img", frame)

    print(count)

