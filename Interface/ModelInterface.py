from abc import ABC, abstractmethod
from keras.api.models import Sequential

class ModelInterface(ABC):
    @abstractmethod
    def setup(
        self,
        datasetsDir: str,
        modelSavePath: str
    ) -> None:
        """
            初始化設定
            datasetsDir: 資料集位置
            modelSavePath: 模型儲存位置
        """
        pass

    @abstractmethod
    def loadModel(self) -> None:
        """
            載入模型
        """
        pass

    @abstractmethod
    def startTraining(self, num_folds, epochs, batch_size) -> None:
        """
            開始訓練
            Argus
                num_folds: kfold 要切幾等份
        """
        pass

    @abstractmethod
    def predict(self, imagePath: str) -> None:
        """
            預測
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
            評估模型
        """
        pass

    @abstractmethod
    def gradCam(self) -> None:
        """
            顯示熱力圖
        """
        pass
