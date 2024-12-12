import json
import os
import cv2
import numpy as np
from glob import glob
import gc
from .ModelInterface import ModelInterface
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.api.models import Sequential
import matplotlib.pyplot as plt
from keras.api.optimizers import Adam
from keras.api.metrics import AUC, Accuracy, F1Score, PrecisionAtRecall
import keras_cv
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


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


    def startTraining(self, num_folds, epochs, batch_size, learning_rate):
        self.batch_size = batch_size

        # 使用 ImageDataGenerator 預處理影像
        datagen = ImageDataGenerator(
            rescale=1.0/255,  # 將像素值歸一化到 [0, 1]
            validation_split=0.2  # 保留 20% 作為測試集
        )

        # 建立數據生成器
        train_generator = datagen.flow_from_directory(
            self.datasetsDir,
            target_size=(self.imageSize[0], self.imageSize[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # KFold 交叉驗證
        kf = KFold(n_splits=num_folds, shuffle=True)

        # 將數據提取為 numpy 數組
        x_data = []
        y_data = []

        for i in range(len(train_generator)):
            x, y = train_generator[i]
            y = np.argmax(y, axis=1)
            y_data = np.expand_dims(y, axis=-1)
            x_data.append(x)
            y_data.append(y)

        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        fold_no = 1
        allHistory = {}
        scores = []
        for train_index, val_index in kf.split(x_data):
            print(f'正在訓練第 {fold_no} 折...')

            # 分割訓練集與驗證集
            X_train, X_val = x_data[train_index], x_data[val_index]
            y_train, y_val = y_data[train_index], y_data[val_index]

            model = self.createModel()

            # 编译模型时确保 metrics 使用正确的参数
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=keras_cv.losses.FocalLoss(gamma=2., alpha=0.25),  # 如果你的输出是概率
                metrics=[
                    AUC(num_thresholds=200, curve="ROC",
                        summation_method="interpolation"),
                    Accuracy(),
                    F1Score(average='micro'),
                    PrecisionAtRecall(0.5, num_thresholds=200)  # 设置适当的 threshold
                ]
            )

            model.summary()

            print("X_train => ", X_train)
            print("Y_train => ", y_train)

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size
            )

            allHistory[fold_no] = history.history

            scores.append(model.evaluate(X_val, y_val, verbose=0))

            # 每次訓練完成後可選擇保存模型
            model.save(f'{self.modelSavePath}/model_fold_{fold_no}.h5')

            print(f'第 {fold_no} 折完成')
            fold_no += 1

            # 清除 gpu 佔用
            tf.keras.backend.clear_session()
            gc.collect()
            # del model, X_train, X_val, y_train, y_val, val_generator
            print(tf.config.experimental.get_memory_info('GPU:0'))


        with open(f'{self.modelSavePath}/training_history.json', 'w') as json_file:
            json.dump(allHistory, json_file, indent=4)
        with open(f'{self.modelSavePath}/training_score.json', 'w') as json_file:
            json.dump(scores, json_file, indent=4)

    def plotTrainingHistory(self):
        # Define file paths
        file_path = f"{self.modelSavePath}/training_history.json"
        output_folder = f"{self.modelSavePath}/plots"
        os.makedirs(output_folder, exist_ok=True)

        # Load the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Define the metrics to plot
        metrics = ["accuracy", "auc", "f1_score", "loss", "precision_at_recall", "val_accuracy", "val_auc", "val_f1_score", "val_loss", "val_precision_at_recall"]

        # Iterate over each metric and save each as an individual image
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for k, values in data.items():
                if metric in values:
                    plt.plot(values[metric], label=f'K={k}')
            plt.title(f"Training Metric: {metric}")
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.legend()
            
            # Save each metric plot as a separate image
            output_path = os.path.join(output_folder, f"{metric}_by_kfold.png")
            plt.savefig(output_path)
            plt.close()

        print(f"All plots have been saved to {output_folder}")


    def predict(self, imagePath: str):
        pass

    def evaluate(self):
        pass

    def gradCam(self):
        pass
