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
vgg16Implement: Interface.ModelInterface = VGG16.ModelImplement() if modelConfig["vgg16"]["enable"] else Interface.BaseModel()
googleNetImplement: Interface.ModelInterface = GoogleNet.ModelImplement() if modelConfig["googleNet"]["enable"] else Interface.BaseModel()
transformerImplement: Interface.ModelInterface = Transformer.ModelImplement() if modelConfig["transformer"]["enable"] else Interface.BaseModel()
denseNetImplement: Interface.ModelInterface = DenseNet.ModelImplement() if modelConfig["denseNet"]["enable"] else Interface.BaseModel()

# 資料統一預處理 ( 如果需要 )
if modelConfig["vgg16"]["traning"]:
    vgg16Implement.startTraining()

if modelConfig["googleNet"]["traning"]:
    googleNetImplement.startTraining()

if modelConfig["transformer"]["traning"]:
    transformerImplement.startTraining()

if modelConfig["denseNet"]["traning"]:
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