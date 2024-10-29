from Interface import BaseModel
from keras.api.layers import (
    Dense, Dropout, Activation, Layer, Embedding, Input, LayerNormalization,
    MultiHeadAttention, Add, Flatten, Resizing, RandomFlip, RandomRotation, RandomZoom
)
from keras.api.metrics import AUC, Accuracy, F1Score, PrecisionAtRecall
from keras.api.models import Model, Sequential
from keras.api.activations import gelu
from keras import backend as K
from tensorflow.keras import mixed_precision
import tensorflow as tf
import keras_cv

mixed_precision.set_global_policy('mixed_float16')

class Patches(Layer):
    def __init__(self, patch_size, batch_size):
        super().__init__()
        self.patch_size = patch_size
        self.batch_size = batch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, (self.batch_size, -1, patch_dims))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(tf.range(start=0, limit=self.num_patches, delta=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config

class ModelImplement(BaseModel):
    def __init__(self):
        super().__init__()

    def createModel(self):
        self.patch_size = 6
        self.num_patches = (self.imageSize[0] // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        self.transformer_layers = 8
        self.mlp_head_units = [
            2048,
            1024,
        ]

        num_classes = 1
        inputs = Input(shape=self.imageSize)
        
        # 数据增强
        augmented = self.dataAugmentation()(inputs)
        
        # 创建补丁
        patches = Patches(self.patch_size, self.batch_size)(augmented)
        
        # 编码补丁
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Transformer 块
        for _ in range(self.transformer_layers):
            # 层归一化 1
            x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
            # 多头注意力层
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # 跳跃连接 1
            x2 = Add()([attention_output, encoded_patches])
            # 层归一化 2
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # 跳跃连接 2
            encoded_patches = Add()([x3, x2])

        # 生成表示
        representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = Flatten()(representation)
        representation = Dropout(0.5)(representation)
        
        # MLP 头部
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        
        # 分类输出
        logits = Dense(num_classes, activation='sigmoid')(features)
        
        # 创建 Keras 模型
        model = Model(inputs=inputs, outputs=logits)
        
        # 编译模型
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

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=gelu)(x)
            x = Dropout(dropout_rate)(x)
        return x

    def dataAugmentation(self):
        data_augmentation = Sequential(
            [
                tf.keras.layers.Normalization(),
                Resizing(self.imageSize[0], self.imageSize[1]),
                RandomFlip("horizontal"),
                RandomRotation(factor=0.02),
                RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )
        return data_augmentation

    def adaptNormalization(self, dataset):
        # 假设 dataset 是一个 tf.data.Dataset 对象，包含训练图像
        data_augmentation = self.dataAugmentation()
        normalization_layer = data_augmentation.layers[0]
        # 提取图像数据并进行适配
        images = dataset.map(lambda x, y: x)
        normalization_layer.adapt(images)
