from abc import ABC, abstractmethod

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
    def startTraining(self) -> None:
        """
            開始訓練
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