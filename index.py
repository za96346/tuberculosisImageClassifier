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

# instance
vgg16Implement: Interface.ModelInterface = VGG16.ModelImplement(
    input_shape=(224,224,3)
) if modelConfig["vgg16"]["enable"] else None
googleNetImplement: Interface.ModelInterface = GoogleNet.ModelImplement(
) if modelConfig["googleNet"]["enable"] else None
transformerImplement: Interface.ModelInterface = Transformer.ModelImplement(
) if modelConfig["transformer"]["enable"] else None
denseNetImplement: Interface.ModelInterface = DenseNet.ModelImplement(
) if modelConfig["denseNet"]["enable"] else None

if modelConfig["vgg16"]["traning"] and vgg16Implement:
    vgg16Implement.setup(
        modelConfig["vgg16"]["datasetsDir"],
        modelConfig["vgg16"]["modelSavePath"]
    )
    vgg16Implement.startTraining(10, 50, 10)

if modelConfig["googleNet"]["traning"] and googleNetImplement:
    googleNetImplement.setup(
        modelConfig["googleNet"]["datasetsDir"],
        modelConfig["googleNet"]["modelSavePath"],
    )
    googleNetImplement.startTraining()

if modelConfig["transformer"]["traning"] and transformerImplement:
    transformerImplement.setup(
        modelConfig["transformer"]["datasetsDir"],
        modelConfig["transformer"]["modelSavePath"],
    )
    transformerImplement.startTraining()

if modelConfig["denseNet"]["traning"] and denseNetImplement:
    denseNetImplement.setup(
        modelConfig["denseNet"]["datasetsDir"],
        modelConfig["denseNet"]["modelSavePath"],
    )
    denseNetImplement.startTraining()

# 使用評估指標評比以上模型
vgg16Implement.evaluate()
googleNetImplement.evaluate()
transformerImplement.evaluate()
denseNetImplement.evaluate()

# 顯示熱力圖
vgg16Implement.gradCam()
googleNetImplement.gradCam()
transformerImplement.gradCam()
denseNetImplement.gradCam()
