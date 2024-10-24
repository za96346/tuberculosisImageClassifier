import json
import Interface
import VGG16
import Transformer
import GoogleNet
import DenseNet

data = {}
with open('./config.json') as f:
    data = json.load(f)

modelConfig = data["models"]

models: dict[str, Interface.ModelInterface] = {
    "VGG16": VGG16.ModelImplement,
    "GoogleNet": GoogleNet.ModelImplement,
    "DenseNet": DenseNet.ModelImplement,
    "Transformer": Transformer.ModelImplement
}

for modelName, modelImplement in models.items():
    thisModelConfig = modelConfig[modelName]
    model: Interface.ModelInterface = modelImplement() if thisModelConfig["enable"] else None

    if modelConfig[modelName]["traning"] and model:
        model.setup(
            thisModelConfig["datasetsDir"],
            thisModelConfig["modelSavePath"],
            (224,224,3),
            (224,224,3)
        )
        model.startTraining(10, 50, 10)

    if model:
        model.evaluate()
        model.gradCam()

