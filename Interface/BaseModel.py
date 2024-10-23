import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from .ModelInterface import ModelInterface
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.api.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class BaseModel(ModelInterface):
    datasetsDir: str  # 資料夾
    modelSavePath: str  # 模型儲存位置
    model: Sequential
    imageSize: tuple[int, int, int]
    normalImageList: list[cv2.typing.MatLike]  # 正常影像matLike list
    tuberculosisImageList: list[cv2.typing.MatLike]  # 肺結核影像matLike list

    def __init__(self, imageSize):
        self.imageSize = imageSize
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

    def setup(self, datasetsDir, modelSavePath):
        self.datasetsDir = datasetsDir
        self.modelSavePath = modelSavePath

    def loadModel(self):
        pass

    # 自定義生成器來根據路徑讀取圖片
    def imageDataGenerator(self, filepaths, labels, batch_size):
        datagen = ImageDataGenerator(rescale=1./255)
        
        while True:
            indices = np.arange(len(filepaths))
            np.random.shuffle(indices)  # 打亂順序
            filepaths = filepaths[indices]
            labels = labels[indices]

            for start in range(0, len(filepaths), batch_size):
                end = min(start + batch_size, len(filepaths))
                batch_paths = filepaths[start:end]
                batch_labels = labels[start:end]
                
                images = []
                for path in batch_paths:
                    img = load_img(path, target_size=self.imageSize)
                    img = img_to_array(img)
                    images.append(img)
                
                images = np.array(images, dtype="float32")
                batch_labels = np.array(batch_labels, dtype="float32").reshape(-1, 1)
                yield images, batch_labels

    def startTraining(self, num_folds, epochs, batch_size):
        # 讀取所有圖片路徑和標籤
        normal_images = glob(os.path.join(self.datasetsDir, 'Normal', '*.png'))
        tb_images = glob(os.path.join(self.datasetsDir, 'Tuberculosis', '*.png'))

        # 建立資料和標籤
        filepaths = normal_images + tb_images
        labels = [0] * len(normal_images) + [1] * len(tb_images)  # 0: Normal, 1: Tuberculosis

        # 將資料轉換為numpy array以便KFold使用
        filepaths = np.array(filepaths)
        labels = np.array(labels)

        # KFold 交叉驗證
        kf = KFold(n_splits=num_folds, shuffle=True)

        fold_no = 1
        for train_index, val_index in kf.split(filepaths):
            print(f'正在訓練第 {fold_no} 折...')

            # 分割訓練集與驗證集
            X_train, X_val = filepaths[train_index], filepaths[val_index]
            y_train, y_val = labels[train_index], labels[val_index]
            
            # 訓練模型
            train_generator = self.imageDataGenerator(X_train, y_train, batch_size)
            val_generator = self.imageDataGenerator(X_val, y_val, batch_size)
            
            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size
            
            self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
            )

            # 每次訓練完成後可選擇保存模型
            self.model.save(f'fold_{fold_no}_{self.modelSavePath}')

            print(f'第 {fold_no} 折完成')
            fold_no += 1

    def predict(self, imagePath: str):
        pass

    def evaluate(self):
        pass

    def gradCam(self):
        pass
