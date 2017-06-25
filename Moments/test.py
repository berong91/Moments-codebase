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
