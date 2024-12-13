from Interface import BaseModel
from keras.api.applications import DenseNet121
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, GlobalAveragePooling2D


class ModelImplement(BaseModel):
    def createModel(self) -> Sequential:
        # 使用 include_top=False 来去除预训练模型的顶层
        base_model = DenseNet121(
            include_top=False,  # 去掉顶层
            weights="imagenet",
            input_shape=self.inputShape,
            pooling=None,
        )

        # 添加自定义的顶层用于二元分类
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),  # Global pooling层替代Flatten，减少参数数量
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        return model
