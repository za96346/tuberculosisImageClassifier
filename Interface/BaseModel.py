import json
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from .ModelInterface import ModelInterface
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.api.models import Sequential
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class BaseModel(ModelInterface):
    datasetsDir: str  # 資料夾
    modelSavePath: str  # 模型儲存位置
    model: Sequential
    imageSize: tuple[int, int, int]
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

    def setup(self, datasetsDir, modelSavePath, imageSize, inputShape):
        self.datasetsDir = datasetsDir
        self.modelSavePath = modelSavePath
        self.imageSize = imageSize
        self.inputShape = inputShape

    def loadModel(self):
        pass

    # 自定義生成器來根據路徑讀取圖片
    def imageDataGenerator(self, filepaths, labels, batch_size):
        datagen = ImageDataGenerator(rescale=1. / 255)

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
                batch_labels = np.array(
                    batch_labels, dtype="float32").reshape(-1, 1)
                yield images, batch_labels

    def startTraining(self, num_folds, epochs, batch_size):
        # 讀取所有圖片路徑和標籤
        normal_images = glob(os.path.join(self.datasetsDir, 'Normal', '*.png'))
        tb_images = glob(
            os.path.join(
                self.datasetsDir,
                'Tuberculosis',
                '*.png'))

        # 建立資料和標籤
        filepaths = normal_images + tb_images
        labels = [0] * len(normal_images) + [1] * \
            len(tb_images)  # 0: Normal, 1: Tuberculosis

        # 將資料轉換為numpy array以便KFold使用
        filepaths = np.array(filepaths)
        labels = np.array(labels)

        # KFold 交叉驗證
        kf = KFold(n_splits=num_folds, shuffle=True)

        fold_no = 1
        allHistory = {}
        for train_index, val_index in kf.split(filepaths):
            print(f'正在訓練第 {fold_no} 折...')

            # 分割訓練集與驗證集
            X_train, X_val = filepaths[train_index], filepaths[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            # 訓練模型
            train_generator = self.imageDataGenerator(
                X_train, y_train, batch_size)
            val_generator = self.imageDataGenerator(X_val, y_val, batch_size)

            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size

            model = self.createModel()

            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
            )

            allHistory[fold_no] = history.history

            # 每次訓練完成後可選擇保存模型
            model.save(f'{self.modelSavePath}/model_fold_{fold_no}.h5')

            print(f'第 {fold_no} 折完成')
            fold_no += 1
        with open(f'{self.modelSavePath}/training_history.json', 'w') as json_file:
            json.dump(allHistory, json_file, indent=4)

    def plotTrainingHistory(self):

        # Load the JSON file data
        file_path = f"{self.modelSavePath}/training_history.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Prepare data for plotting (epoch 50, kfold k=10)
        epochs = list(range(1, 51))  # 50 epochs
        accuracy = data[0]['1']['accuracy'][:50]
        auc = data[0]['1']['auc'][:50]
        f1_score = data[0]['1']['f1_score'][:50]
        loss = data[0]['1']['loss'][:50]
        val_accuracy = data[0]['1']['val_accuracy'][:50]
        val_auc = data[0]['1']['val_auc'][:50]
        val_f1_score = data[0]['1']['val_f1_score'][:50]
        val_loss = data[0]['1']['val_loss'][:50]

        # Create the plot
        plt.figure(figsize=(10, 8))

        plt.plot(epochs, accuracy, label="Accuracy")
        plt.plot(epochs, auc, label="AUC")
        plt.plot(epochs, f1_score, label="F1 Score")
        plt.plot(epochs, loss, label="Loss")
        plt.plot(epochs, val_accuracy, label="Val Accuracy", linestyle='--')
        plt.plot(epochs, val_auc, label="Val AUC", linestyle='--')
        plt.plot(epochs, val_f1_score, label="Val F1 Score", linestyle='--')
        plt.plot(epochs, val_loss, label="Val Loss", linestyle='--')

        plt.title("Training and Validation Metrics Over 50 Epochs (K=10)")
        plt.xlabel("Epoch")
        plt.ylabel("Metrics")
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()


    def predict(self, imagePath: str):
        pass

    def evaluate(self):
        pass

    def gradCam(self):
        pass
