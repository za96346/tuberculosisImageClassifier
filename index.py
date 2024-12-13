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
    model: Interface.ModelInterface = modelImplement(
    ) if thisModelConfig["enable"] else None

    if model:
        model.setup(
            thisModelConfig["datasetsDir"],
            thisModelConfig["modelSavePath"],
            (224, 224, 3),
            (224, 224, 3)
        )

        if modelConfig[modelName]["traning"] and model:
            model.startTraining(5, 20, 64, 0.001)

        if modelConfig[modelName]["plotTrainingHistory"]:
            model.plotTrainingHistory()

        model.evaluate()
        model.gradCam()

