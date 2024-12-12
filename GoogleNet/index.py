from Interface import BaseModel
from keras.api.applications import InceptionV3
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, GlobalAveragePooling2D

class ModelImplement(BaseModel):
    def createModel(self) -> Sequential:
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=self.inputShape
        )
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),  # Global pooling层替代Flatten，减少参数数量
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        return model
