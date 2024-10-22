import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import random
import cv2
from tqdm import tqdm
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# gpu check
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 可選：設置為按需使用內存
    except RuntimeError as e:
        print(e)

# load data
no_tb_data = "./TB_Chest_Radiography_Database/Normal"
tb_data = "./TB_Chest_Radiography_Database/Tuberculosis"

# resize image and let it to matLike


def imagePreprocess(imageFolderPath: str):
    data = []
    tq = tqdm(os.listdir(imageFolderPath))
    tq.set_description(f"圖像預處理 ({imageFolderPath}): ")
    for image in tq:
        image_path = os.path.join(imageFolderPath, image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        data.append(img)

    return data


tb_image = np.array(imagePreprocess(tb_data))
no_image = imagePreprocess(no_tb_data)

# tb 資料處理
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

aug_images = []
for image in tqdm(tb_image):
    image = np.expand_dims(image, axis=0)
    i = 0
    for batch in datagen.flow(image, batch_size=1):
        aug_images.append(batch[0])
        i += 1
        if i >= 5:
            break

TB_yes = []
for image in tqdm(aug_images):
    TB_yes.append([image, 1])

# normal 資料處理
TB_no = []
for image in tqdm(no_image):
    TB_no.append([image, 0])

# 資料集分組
data = TB_yes + TB_no
random.shuffle(data)

X = []
y = []
for i, j in tqdm(data):
    X.append(i)
    y.append(j)

x = np.array(X)
y = np.array(y)

# train data
x_train = x[:5500]
y_train = y[:5500]

# test data
x_test = x[5500:7000]
y_test = y[5500:7000]

# model 模型建構
model = Sequential()
model.add(Conv2D(100, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(100, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
# model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(.2))
# model.add(Dense(32, activation = "relu"))
model.add(Dropout(.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation='sigmoid'))


model.compile(
    optimizer="adam",
    loss='binary_crossentropy',
    metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=.2, epochs=5)


# loss and accuracy plot
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()


# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()

model.save('my_vgg16_model.h5')

model.evaluate(x_test, y_test)
