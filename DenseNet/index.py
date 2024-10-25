from Interface import BaseModel
from keras.api.applications import DenseNet121
from keras.api.models import Sequential
from keras.api.metrics import AUC, Accuracy, F1Score, PrecisionAtRecall
from keras.api.layers import Dense, Activation, Dropout, Flatten, GlobalAveragePooling2D
import keras_cv


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
            Dense(1, activation='sigmoid')  # 二元分类的输出
        ])

        # 编译模型时确保 metrics 使用正确的参数
        model.compile(
            optimizer='adam',
            loss=keras_cv.losses.FocalLoss(from_logits=False),  # 如果你的输出是概率
            metrics=[
                AUC(num_thresholds=200, curve="ROC",
                    summation_method="interpolation"),
                Accuracy(),
                F1Score(average='micro'),  # 二元分类
                PrecisionAtRecall(0.5, num_thresholds=200)  # 设置适当的 threshold
            ]
        )

        model.summary()

        return model
