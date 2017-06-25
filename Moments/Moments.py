from src.Analysis.Model import Model 

class Moments :
    def __init__(self, *args, **kwargs) :
        print("init model")
        self.model = Model()
        print("creating model")
        self.model.CreateModel()
        print("training model")
        self.model.TrainModel()

    def ClassifyImage(self, image) :
        pass

    def GetFaces(self, image) :
        pass

    def ClassifyFace(self, image) :
        pass
    
    def StoreImage(self, image, directory) :
        pass

def GenerateDataset() :
    import pandas as pd
    from PIL import Image
    import numpy as np
    import math
    from matplotlib import pyplot
    import scipy.misc as misc
    num = 0
    for file in pd.read_csv("train.csv",  chunksize = 1) :
        try :
            data = np.array([int(x) for x in file["Pixels"].iloc[0].split()])
            data = np.resize(data, (48,48))
            misc.imsave("faces/image{0}_{1}.png".format(num, file["Emotion"].iloc[0]), data)
            print(num)
            num+=1
        except Exception as e:
            print(e)
            break

moments = Moments()