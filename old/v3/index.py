import numpy as np  # linear algebra
import pandas as pd
import os
import cv2
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from keras import layers, Sequential, models, losses
import tensorflow as tf

# gpu check
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 可選：設置為按需使用內存
    except RuntimeError as e:
        print(e)

# Define path to the data directory
data_dir = "./TB_Chest_Radiography_Database"

# Get the path to the normal and pneumonia sub-directories
normal_cases_dir = data_dir + '/Normal'
Tuberculosis_cases_dir = data_dir + '/Tuberculosis'

# Get the list of all the images
normal_cases = [
    f"{normal_cases_dir}/{i}" for i in os.listdir(normal_cases_dir)]
Tuberculosis_cases = [
    f"{Tuberculosis_cases_dir}/{i}" for i in os.listdir(Tuberculosis_cases_dir)]

train_data = []
train_labels = []


def transImage(label, images):
    tq = tqdm(images)
    tq.set_description(f"[{label}]圖像轉換成matrix")
    for img in tq:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224, 224))
        if img.shape[2] == 1:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = img / 255

        train_data.append(img)
        train_labels.append(label)


transImage("normal", normal_cases)
transImage("Tuberculosis", Tuberculosis_cases)

# Convert the list into numpy arrays
train_data1 = np.array(train_data)
train_labels1 = np.array(train_labels)

print("Total number of validation examples: ", train_data1.shape)
print("Total number of labels:", train_labels1.shape)

train_labels1 = pd.DataFrame(train_labels1, columns=['label'], index=None)
print(train_labels1.head(), train_labels1.tail())

train_labels1['label'] = train_labels1['label'].map(
    {'normal': 0, 'Tuberculosis': 1})
print(train_labels1['label'].unique())

# 處理資料平衡
smt = SMOTE()
train_rows = len(train_data1)
train_data1 = train_data1.reshape(train_rows, -1)
train_data2, train_labels2 = smt.fit_resample(train_data1, train_labels1)

cases_count1 = train_labels2['label'].value_counts()

train_data2 = train_data2.reshape(-1, 224, 224, 3)

X_train, X_test, y_train, y_test = train_test_split(
    train_data2, train_labels2, test_size=0.13, random_state=42)

# 模型建構
data_augmentation = Sequential(
    [
        layers.RandomFlip(
            "horizontal",
            input_shape=(224, 224, 3)
        ),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])


model.summary()

# complie and train
model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.fit(
    np.array(X_train),
    np.array(y_train),
    epochs=200,
    validation_data=(np.array(X_test), np.array(y_test))
)

model.save("TuberculosisImageClassifier.keras")
