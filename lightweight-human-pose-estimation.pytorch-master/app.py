#!/usr/bin/env python
from flask import Flask, render_template, Response
import io
import cv2

import pickle
import socket
import struct
import torch
import cv2
import os

from test import run_demo, ImageReader
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

import cv2
import numpy as np
import socket
import sys
import pickle
import struct
from flask import Flask, render_template, Response, jsonify
import io
#import voice

app = Flask(__name__)
count = 0

@app.route('/count')
def getCount():
    global count

    return jsonify(count=count)

@app.route('/<type>')
def index(type):
    """Video streaming home page."""
    if type == 'squat':
        return render_template('squat.html')
    elif type == 'pushup':
        return render_template('pushup.html')


def gen():
    """Video streaming generator function."""
    HOST = ''
    PORT = 8088

    emptyPoses = []

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)


    conn, addr = s.accept()
    print('ACCENPTED')

    data = b''  ### CHANGED
    payload_size = struct.calcsize("=L")  ### CHANGED

    stepA = False
    stepB = False
    global count

    sitAngle = 0
    stdupAngle = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 400)
    topLeft = (150, 400)

    fontScale = 3
    fontColor = (255, 0, 0)
    lineType = 2

    emptyPoses = []

    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
        print('MESSAGESIZE')
        print(payload_size)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("=L", packed_msg_size)[0]  ### CHANGED
        print('unpack')
        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)
            print(len(data))
            print(msg_size)
        print('RECIEVED')

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)

        #read_return_code, frame = vc.read()

        pose = run_demo(net, frame, 256, 0, 0, 1)

        if pose is not None:
            pose.draw(frame)
            #cv2.imshow('test', frame)
            if cv2.waitKey(1) == ord('q'):
                break

            A = np.array([pose.keypoints[8][0], pose.keypoints[8][1]])
            B = np.array([pose.keypoints[9][0], pose.keypoints[9][1]])
            C = np.array([pose.keypoints[10][0], pose.keypoints[10][1]])
    
            BA = A - B
            BC = C - B

            cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            angle = np.arccos(cosine_angle)
            angle = np.degrees(angle)
            #print(angle)

            if angle > 140:
                stepA = True
                if stdupAngle is 0:
                    stdupAngle = angle
                if angle > stdupAngle:
                    stdupAngle = angle
            if angle < 70:
                stepB = True
                if sitAngle is 0:
                    sitAngle = angle
                if angle < sitAngle:
                    sitAngle = angle

            if stepA and stepB is True:
                if angle > 140:
                    stepA = False
                    stepB = False
                    count += 1

                    if sitAngle > 60:
                        cv2.putText(frame, "Bend your knee more",
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
            if stepA is True and stepB is False:
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
            #cv2.imshow("img", frame)

            print(count)
            encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
            io_buf = io.BytesIO(image_buffer)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


def gen2():
    """Video streaming generator function."""
    HOST = ''
    PORT = 8088

    emptyPoses = []

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)


    conn, addr = s.accept()
    print('ACCENPTED')

    data = b''  ### CHANGED
    payload_size = struct.calcsize("=L")  ### CHANGED

    A = 0
    B = 0
    C = 0

    stepA = False
    stepB = False
    global count

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 400)
    topLeft = (150, 400)

    fontScale = 3
    fontColor = (255, 0, 0)
    lineType = 2

    emptyPoses = []

    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
        print('MESSAGESIZE')
        #print(payload_size)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("=L", packed_msg_size)[0]  ### CHANGED
        print('unpack')
        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)
            #print(len(data))
            #print(msg_size)
        print('RECIEVED')

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)

        #read_return_code, frame = vc.read()

        pose = run_demo(net, frame, 256, 0, 0, 1)

        if pose is not None:
            pose.draw(frame)
        #cv2.imshow('test', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        A = np.array([pose.keypoints[2][0], pose.keypoints[2][1]])
        if B is 0:
            B = np.array([pose.keypoints[10][0], pose.keypoints[10][1]])
        if C is 0:
            C = np.array([pose.keypoints[4][0], pose.keypoints[4][1]])

        BA = A - B
        BC = C - B

        cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        #print(angle)

        if angle > 25:
            stepA = True
        if angle < 15:
            stepB = True

        if stepA and stepB is True:
            if angle > 25:
                stepA = False
                stepB = False
                count += 1

                cv2.putText(frame, "good",
                            topLeft,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
        if stepA is True and stepB is False:
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

        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'

    )

@app.route('/video_feed2')
def video_feed2():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen2(),
        mimetype='multipart/x-mixed-replace; boundary=frame'

    )

if __name__ == '__main__':
    #s.close()
    app.run(host='0.0.0.0', debug=True, threaded=True)

