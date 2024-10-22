import os
import cv2
from tqdm import tqdm
from .ModelInterface import ModelInterface
import tensorflow as tf


class BaseModel(ModelInterface):
    datasetsDir: str  # 資料夾
    modelSavePath: str  # 模型儲存位置
    normalImageList: list[cv2.typing.MatLike]  # 正常影像matLike list
    tuberculosisImageList: list[cv2.typing.MatLike]  # 肺結核影像matLike list

    def __init__(self):
        self.__useGPU__()
        super().__init__()

    # 使用gpu
    def __useGPU__(self):
        print(
            "Num GPUs Available: ", len(
                tf.config.list_physical_devices('GPU')))

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print(e)

    # resize image and let it to matLike
    def __imageToMatLike__(self,
                           imageFolderPath: str) -> list[cv2.typing.MatLike]:
        data = []
        tq = tqdm(os.listdir(imageFolderPath))
        tq.set_description(f"圖像預處理 ({imageFolderPath}): ")
        for image in tq:
            image_path = os.path.join(imageFolderPath, image)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            data.append(img)

        return data

    def setup(self, datasetsDir, modelSavePath):
        self.datasetsDir = datasetsDir
        self.modelSavePath = modelSavePath

        self.normalImageList = self.__imageToMatLike__(
            f"{self.datasetsDir}/Normal")
        self.tuberculosisImageList = self.__imageToMatLike__(
            f"{self.datasetsDir}/Tuberculosis")

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
