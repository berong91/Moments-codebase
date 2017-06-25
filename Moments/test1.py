import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import h5py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.models import load_model
import PIL
import cv2
from PIL import Image

# (beween 0 and 6: anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6)
labels = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    facebox = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        facebox.append((x, y, x + w, y + h))

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == 32:
        for b in facebox:
            face = Image.fromarray(img)
            face = face.crop(b)
            face = face.convert('L')
            face = face.resize((48, 48), PIL.Image.BICUBIC)

            pred_x = np.reshape(face, (1, 48, 48, 1))
            p = model.predict(pred_x)
            print(p)
            print(labels[np.argmax(p)])
