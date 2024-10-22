import os
import tensorflow as tf
from keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.applications import VGG16
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 可選：設置為按需使用內存
    except RuntimeError as e:
        print(e)

data_dir = 'TB_Chest_Radiography_Database/'  # 数据主目录
classes = ['Normal', 'Tuberculosis']  # 类别名
image_paths = []
labels = []

# 遍历文件夹，收集图像路径和对应标签
for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for file_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, file_name))
        labels.append(label)

# 划分为训练集和测试集
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

# 进一步将训练集划分为训练集和验证集
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, random_state=42, stratify=train_labels)  # 验证集占 25%


# 创建数据增强器
train_datagen = ImageDataGenerator(
    rescale=1. / 255,           # 归一化像素值
    rotation_range=40,        # 数据增强参数
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 验证和测试集不做数据增强，只进行归一化
val_test_datagen = ImageDataGenerator(rescale=1. / 255)


def image_generator(image_paths, labels, batch_size, datagen):
    while True:
        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_paths = image_paths[start:end]
            batch_labels = labels[start:end]

            # 加载图像并应用预处理
            batch_images = [
                tf.keras.preprocessing.image.load_img(
                    img_path, target_size=(
                        224, 224)) for img_path in batch_paths]
            batch_images = np.array(
                [tf.keras.preprocessing.image.img_to_array(img) for img in batch_images])

            # 使用 ImageDataGenerator 对图像进行处理
            batch_images = datagen.flow(
                batch_images,
                batch_size=len(batch_images),
                shuffle=False)[0]

            yield batch_images, np.array(batch_labels)


batch_size = 32

# 生成器
train_generator = image_generator(
    train_paths,
    train_labels,
    batch_size,
    train_datagen)
val_generator = image_generator(
    val_paths,
    val_labels,
    batch_size,
    val_test_datagen)
test_generator = image_generator(
    test_paths,
    test_labels,
    batch_size,
    val_test_datagen)

# 假设你已经定义好了模型
steps_per_epoch_train = len(train_paths) // batch_size
steps_per_epoch_val = len(val_paths) // batch_size
steps_per_epoch_test = len(test_paths) // batch_size

# 加载 VGG16 模型，去掉顶层全连接层 (include_top=False)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(
        224,
        224,
        3))

# 冻结 VGG16 的卷积层
base_model.trainable = False

# 创建自己的模型，在 VGG16 基础上添加全连接层
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))  # 假设是二分类

# 打印模型摘要
model.summary()

# 编译模型
model.compile(optimizer='adam',              # 优化器
              loss='sparse_categorical_crossentropy',  # 损失函数，适用于多分类问题
              metrics=['accuracy'])          # 评估指标

# 开始训练
history = model.fit(
    train_generator,                         # 训练数据生成器
    steps_per_epoch=steps_per_epoch_train,   # 每个 epoch 的步骤数
    validation_data=val_generator,           # 验证数据生成器
    validation_steps=steps_per_epoch_val,    # 验证集的步骤数
    epochs=50                                # 训练的 epoch 数
)

# 保存模型
model.save('my_vgg16_model.h5')

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(
    test_generator, steps=steps_per_epoch_test)
print(f"Test accuracy: {test_acc}")
