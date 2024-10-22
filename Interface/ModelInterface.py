from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def __init__(
        datasetsDir: str,
        modelSavePath: str
    ) -> None:
        super().__init__()
        pass

    """
        載入模型
    """
    @abstractmethod
    def loadModel() -> None:
        pass

    """
        開始訓練
    """
    @abstractmethod
    def startTraining() -> None:
        pass