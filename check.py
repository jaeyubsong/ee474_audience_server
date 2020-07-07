from utils import *
net = cv2.dnn.readNetFromDarknet('./cfg/yolov3-face.cfg', './model-weights/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#from flask import Flask, Response, render_template, request, jsonify
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#app = Flask(__name__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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

    if CRR_FRAME > NUM_FRAME:
        return PREVIOUS_RESULTS

    elif CRR_FRAME <= NUM_FRAME:
        
        for i in range(len(frame_list)):
            cv2.imwrite('frame/frame/'+str(i)+ '.png', frame_list[i])

        test_generator = get_datagen('frame', len(frame_list))
        output= model.predict_generator(test_generator, steps=1)

        ychats = np.array(output)
        summed = np.sum(ychats, axis=0)
        outcomes = np.argmax(ychats, axis=1)
        #frame_list = []
        CRR_FRAME = 0
        print(outcomes)
        print(summed)
        if statistics:
            results = np.array([0, 0, 0, 0, 0], dtype=np.float32)

            for out in outcomes:
                results[out] += 1

            results = np.divide(summed, len(frame_list))
            results = np.multiply(results, 100)
            PREVIOUS_RESULTS = results
            frame_list = []
            return results

image = cv2.imread('zoom.png', cv2.IMREAD_COLOR)
cv2.imwrite('zoom1.png', image)
frame = cv2.resize(image, dsize=(416, 416))
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
    cv2.imwrite(str(j)+'.jpg', temp)
    crop_faces.append(temp) 
for i in range(9):
    image = cv2.imread(str(i)+'.png', cv2.IMREAD_COLOR)
    crop_faces.append(image)
print(total_faces)
print('#'*60)
statistics= get_emotion(crop_faces)
emotions = ['astonished', 'unsatisfied', 'joyful', 'sad', 'neutral']
print(emotions)
print(statistics)
print(total_faces)

