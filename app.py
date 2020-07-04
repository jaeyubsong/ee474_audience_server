from flask import Flask, Response, render_template, request, jsonify
import logging
from config import *
import numpy as np
import cv2
import json
import random
import argparse
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

app = Flask(__name__)


# Get statistics
def get_audience_statistics(image):
    # Implement code for getting statistics
    time.sleep(0.3)
    statistics = [random.random() for i in range(5)]
    statistics_sum = sum(statistics)
    statistics = [ int(100*i/statistics_sum) for i in statistics ]
    total_faces = random.randint(5, 10)
    drowsy_faces = random.randint(1,3)
    return statistics, total_faces, drowsy_faces



# Global variabble
SURGICAL_MASK = 1
SHOWMASK = True
CUR_MASK = SURGICAL_MASK

@app.route('/')
def index():
    return "Hi there"


@app.route('/audienceInfo', methods=['POST'])
def emotion():
    print("Emotion")
    # npimg = np.fromstring(request.files['file'].read(), np.uint8)
    npimg = np.frombuffer(request.files['file'].read(), np.uint8)
    # data = json.loads(request.form['json'])
    # landmark = data['landmark']
    # print("get landmark")
    # print(landmark)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite("saved.jpg", img)
    print(img.shape)
    ##############################
    # get_audience_statistics should return 'statistics' in list with length of 5, 'total_faces' in int, and drowsy_faces in int.
    statistics, total_faces, drowsy_faces = get_audience_statistics(img)
    # To the preprocessing here
    ##############################
    response = jsonify({'astonished': statistics[0], 'unsatisfied': statistics[1], 
                        'joyful': statistics[2], 'neutral': statistics[3], 
                        'sad': statistics[4], 'total_faces': total_faces, 'drowsy_faces': drowsy_faces})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    return response

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", required=False, type=int, default=7008, help="Write your desired port number")
    args = vars(ap.parse_args())
    app.run(host='0.0.0.0', port=args['port'])