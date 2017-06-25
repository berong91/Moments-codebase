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

#(beween 0 and 6: anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6)
labels = {0 : "anger" , 1 : "disgust" ,  2 : "fear", 3 : "happy", 4 : "sad" , 5 : "surprise" ,  6 : "neutral"}

data = pd.read_csv("train.csv")

im = data['Pixels']
img = []
for i in range(len(im)):
	img.append(list(map(int, im[i].split())))
	
img = np.asarray(img).astype(np.float)


x_train = np.reshape(img, (4178, 48, 48, 1))
y_train = np_utils.to_categorical(data["Emotion"])

pred_x = x_train[0,:,:,0]

pred_x = np.reshape(pred_x,(1,48,48,1))
model = load_model("model.h5")
p = model.predict(pred_x)
print(p)
print(labels[np.argmax(p)])