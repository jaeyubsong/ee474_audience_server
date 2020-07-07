from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import logging
from config import *
import numpy as np
import cv2
import json
import random
import argparse
import os
import time
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

CUR_COMBINED_IMG = None

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

app = Flask(__name__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def resize_image(inputImg, width=None, height=None):
    org_width, org_height, _ = inputImg.shape
    if width is None and height is None:
        # # printt("Error as both width and height is NONE")
        exit(-1)
    elif width is None:
        ratio = 1.0 * height / org_height
        width = int(ratio * org_width)
    elif height is None:
        ratio = 1.0 * width / org_width
        height = int(ratio * org_height)
    dim = (int(width), int(height))
    # # printt(dim)
    resized = cv2.resize(inputImg, dim, interpolation=cv2.INTER_AREA)
    return resized




def get_datagen(dataset, num, aug=False):
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
            dataset,
            target_size=(197, 197),
            color_mode='rgb',
            shuffle = False,
            class_mode='categorical',
            batch_size=num)

model = load_model("./models/cnn.h5")
NUM_FRAME = 30
frame_list = []
CRR_FRAME = 0
PREVIOUS_RESULTS = np.array([0, 0, 0, 0, 0], dtype=np.float32)
# Get statistics
def get_emotion(img_list, statistics = True):
    global frame_list
    global CRR_FRAME
    global PREVIOUS_RESULTS

    for img in img_list:
        frame_list.append(img)
    CRR_FRAME += 1
    emotions = ['astonished', 'unsatisfied', 'joyful', 'sadness', 'neutral']

    if CRR_FRAME < NUM_FRAME:
        return PREVIOUS_RESULTS

    elif CRR_FRAME >= NUM_FRAME:
        
        for i in range(len(frame_list)):
            cv2.imwrite('frame/frame/'+str(i)+ '.png', frame_list[i])

        test_generator = get_datagen('frame', len(frame_list))
        output= model.predict_generator(test_generator, steps=1)

        ychats = np.array(output)
        summed = np.sum(ychats, axis=0)
        outcomes = np.argmax(ychats, axis=1)
        frame_list = []
        CRR_FRAME = 0

        if statistics:
           
            results = np.divide(summed, NUM_FRAME)
            results = np.multiply(results, 100)
            PREVIOUS_RESULTS = results

            return results

# Global variabble
SURGICAL_MASK = 1
SHOWMASK = True
CUR_MASK = SURGICAL_MASK

@app.route('/')
def index():
    return "Hi there"


from utils import *
net = cv2.dnn.readNetFromDarknet('./cfg/yolov3-face.cfg', './model-weights/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def getCombinedImg():
    global CUR_COMBINED_IMG
    while True:
        time.sleep(0.05)
        # CUR_COMBINED_IMG = cv2.imread('./zoom.png', cv2.IMREAD_UNCHANGED)
        if CUR_COMBINED_IMG is None:
            return
        ret, jpeg = cv2.imencode('.jpg', CUR_COMBINED_IMG)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/getAudienceCombined', methods=['GET'])
def stream():
    return Response(stream_with_context(getCombinedImg()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/audienceInfo', methods=['POST'])
def emotion():
    global CUR_COMBINED_IMG
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
    drowsy_faces = 0
    ############################## 
    # yolo face detection: 'total_faces' in int
    frame = cv2.resize(img, dsize=(416, 416))
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    total_faces = len(faces)
    crop_faces = []
    for j, face in enumerate(faces):
        left, top, w, h = face
        temp = frame.copy()
        temp = temp[top:top+h, left:left+w]
        crop_faces.append(temp) 
    # Code to append the images
    if total_faces > 0:
        first_img = crop_faces[0]
        first_img = resize_image(first_img, width=150, height=200)
        resultImg = first_img
        if total_faces > 5:
            tmpImg = crop_faces[5]
            tmpImg = resize_image(tmpImg, width=150, height=200)
            down_img = tmpImg
        for i in range(1, total_faces):
            if i == 5 and i != total_faces-1:
                continue
            tmpImg = crop_faces[i]
            tmpImg = resize_image(tmpImg, width=150, height=200)
            if i < 5:
                resultImg = np.hstack((resultImg, tmpImg))
            elif i >= 5 and i != total_faces - 1:
                down_img = np.hstack((down_img, tmpImg))
            elif i >= 5 and i == total_faces - 1:
                down_img = np.hstack((down_img, tmpImg))
                npad = ((0, 0), (0, 150*(9-i)), (0,0))
                down_img = np.pad(down_img, npad, 'constant', constant_values = (0))
                resultImg = np.vstack((resultImg, down_img))
                CUR_COMBINED_IMG = resultImg
    ##############################
    # get_audience_statistics should return 'statistics' in list with length of 5 
    if total_faces > 0:
        statistics= get_emotion(crop_faces)
    else:
        statistics = [0, 0, 0, 0, 100]
    # To the preprocessing here
    ##############################
    #drowsy_faces in int.

    ##############################

    response = jsonify({'astonished': statistics[0], 'unsatisfied': statistics[1], 
                        'joyful': statistics[2], 'sad': statistics[3], 
                        'neutral': statistics[4], 'total_faces': total_faces, 'drowsy_faces': drowsy_faces})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    return response

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", required=False, type=int, default=7008, help="Write your desired port number")
    args = vars(ap.parse_args())
    app.run(host='0.0.0.0', port=args['port'])
