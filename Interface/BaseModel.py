from .ModelInterface import ModelInterface

class BaseModel(ModelInterface):
    datasetsDir: str
    modelSavePath: str
    def __init__(self):
        super().__init__()
    
    def setup(self, datasetsDir, modelSavePath):
        self.datasetsDir = datasetsDir
        self.modelSavePath = modelSavePath
    
    def loadModel(self):
        pass

    def startTraining(self):
        pass

    def predict(self, imagePath: str):
        pass

    def evaluate(self):
        pass

    def gradCam(self):
        pass