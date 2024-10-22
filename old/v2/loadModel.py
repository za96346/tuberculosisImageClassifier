from keras.api.models import load_model
from keras.api.applications.vgg16 import preprocess_input, decode_predictions
from keras.api.preprocessing import image
from keras.api.layers import Lambda
import keras.api.backend as K
import tensorflow as tf
import numpy as np
import cv2
import sys

def load_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


model_path = "my_vgg16_model.h5"  # 替換成你的自定義模型文件路徑
model = load_model(model_path)

# Load image and preprocess
preprocessed_input = load_image(sys.argv[1])

# Predict
predictions = model.predict(preprocessed_input)
print(predictions)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class} with confidence {predictions[0][predicted_class]:.2f}")