from Interface import BaseModel
from keras.api.models import Sequential
from keras.api.metrics import AUC, Accuracy, F1Score, PrecisionAtRecall
from keras.api.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import keras_cv


class ModelImplement(BaseModel):
    def createModel(self) -> Sequential:
        model = Sequential([
            Conv2D(64, (3, 3), input_shape=self.inputShape, padding='same',
                activation='relu'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # 編譯模型時確保 metrics 使用正確的參數
        model.compile(
            optimizer='adam',
            loss=keras_cv.losses.FocalLoss(from_logits=False),  # 如果你的輸出是概率
            metrics=[
                AUC(
                    num_thresholds=200,
                    curve="ROC",
                    summation_method="interpolation",
                ),
                Accuracy(),
                F1Score(average='micro'),  # 適合二元分類
                PrecisionAtRecall(0.5, num_thresholds=200)  # 設定適當的 threshold
            ]
        )


        model.summary()
        
        return model

