from Model import Model
class EmotionDetector:
    
    def __init__(self) :
        self.model = Model()

    def Train(self, dataset) :
        self.model.CreateModel()


    def Pickle(self) :
        pass

