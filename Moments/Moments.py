

class Moments :
    def __init__(self, *args, **kwargs) :
        pass

    def ClassifyImage(self, image) :
        pass

    def GetFaces(self, image) :
        pass

    def ClassifyFace(self, image) :
        pass
    
    def StoreImage(self, image, directory) :
        pass


import pandas as pd
from PIL import Image
import numpy as np

for file in pd.read_csv("train.csv", sep=",", chunksize = 1) :
    data = np.array([int(x) for x in file.get("Pixels")[0].split()])    
    data = np.reshape(data, (48,48))
    img = Image.fromarray(data)
    img.show()
    break