import json
import os
import cv2
import numpy as np
from glob import glob
import gc
import pandas as pd
from .ModelInterface import ModelInterface
import tensorflow as tf
from keras.api.models import Sequential
import matplotlib.pyplot as plt
from keras.api.optimizers import Adam
from keras.api.metrics import AUC, Accuracy, F1Score, PrecisionAtRecall
import keras_cv
from sklearn.model_selection import StratifiedKFold
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

    def imagePreprocess(self):
        # 掃描目錄中的圖像路徑和標籤
        all_image_paths = []
        all_labels = []

        for class_name in os.listdir(self.datasetsDir):
            class_dir = os.path.join(self.datasetsDir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    all_image_paths.append(img_path)
                    all_labels.append(class_name)

        all_image_paths = np.array(all_image_paths)
        all_labels = np.array(all_labels)

        print("all_image_paths length => ", len(all_image_paths))
        print("all_labels length => ", len(all_labels))
        
        print("all_image_paths =>", all_image_paths)
        print("all_labels =>", all_labels)

        return all_image_paths, all_labels


    def startTraining(self, num_folds, epochs, batch_size, learning_rate):
        self.batch_size = batch_size

        all_image_path, all_labels = self.imagePreprocess()

        # KFold 交叉驗證
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        allHistory = {}
        for  fold, (train_index, val_index) in enumerate(skf.split(all_image_path, all_labels)):
            print(f'正在訓練第 {fold} 折...')

            # 分割訓練集與驗證集
            train_paths, val_paths = all_image_path[train_index], all_image_path[val_index]
            train_labels, val_labels = all_labels[train_index], all_labels[val_index]

            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            train_generator = datagen.flow_from_dataframe(
                dataframe=pd.DataFrame({'filename': train_paths, 'class': train_labels}),
                x_col='filename',
                y_col='class',
                target_size=(self.imageSize[0], self.imageSize[1]),
                batch_size=batch_size,
                class_mode='categorical'
            )
            
            val_generator = datagen.flow_from_dataframe(
                dataframe=pd.DataFrame({'filename': val_paths, 'class': val_labels}),
                x_col='filename',
                y_col='class',
                target_size=(self.imageSize[0], self.imageSize[1]),
                batch_size=batch_size,
                class_mode='categorical'
            )

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

            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                batch_size=batch_size
            )

            allHistory[fold] = history.history

            # 每次訓練完成後可選擇保存模型
            model.save(f'{self.modelSavePath}/model_fold_{fold}.h5')

            print(f'第 {fold} 折完成')

            # 清除 gpu 佔用
            tf.keras.backend.clear_session()
            gc.collect()
            # del model, X_train, X_val, y_train, y_val, val_generator
            print(tf.config.experimental.get_memory_info('GPU:0'))


        with open(f'{self.modelSavePath}/training_history.json', 'w') as json_file:
            json.dump(allHistory, json_file, indent=4)

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
