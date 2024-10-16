import tensorflow as tf
from keras.api.models import load_model
import numpy as np
import cv2

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 可选：设置为按需使用内存
    except RuntimeError as e:
        print(e)

model = load_model("./TuberculosisImageClassifier.keras")
model.summary()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # 添加批次维度
    return img

preprocessed_image = preprocess_image("/Users/siou/Downloads/code/TuberculosisImageClassifier/TB_Chest_Radiography_Database/Normal/Normal-2.png")

predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions, axis=1)

# 映射回标签
label_map = {0: 'normal', 1: 'Tuberculosis'}
print(f"预测结果: {label_map[predicted_class[0]]}")