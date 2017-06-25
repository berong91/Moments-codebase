from Model import Model
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
class EmotionDetector:
    
    def __init__(self) :
        self.model = Model()

    def Train(self, dataset) :
        self.model.CreateModel()


    def Pickle(self) :
        pass

