from keras.api.models import load_model
from keras.api.applications.vgg16 import preprocess_input, decode_predictions
from keras.api.preprocessing import image
from keras.api.layers import Lambda
import keras.api.backend as K
import tensorflow as tf
import numpy as np
import cv2
import sys

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def normalize(x):
    # Normalizes a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 2  # 假設你訓練的是二分類
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    
    with tf.GradientTape() as tape:
        conv_output = [l for l in input_model.layers if l.name == layer_name][0].output
        loss = K.sum(target_layer(conv_output))
        grads = tape.gradient(loss, conv_output)
    
    grads = normalize(grads[0])
    conv_output = conv_output[0].numpy()
    grads = grads.numpy()

    weights = np.mean(grads, axis=(0, 1))
    cam = np.ones(conv_output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image[0])
    cam = 255 * cam / np.max(cam)
    
    return np.uint8(cam), heatmap

def deprocess_image(x):
    x = np.squeeze(x)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # Clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Load your custom VGG16 model
model_path = "my_vgg16_model.h5"  # 替換成你的自定義模型文件路徑
model = load_model(model_path)

# Load image and preprocess
preprocessed_input = load_image(sys.argv[1])

# Predict
predictions = model.predict(preprocessed_input)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class} with confidence {predictions[0][predicted_class]:.2f}")

# Generate Grad-CAM
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
cv2.imwrite("gradcam.jpg", cam)

# Save Guided Grad-CAM
gradcam = heatmap[..., np.newaxis]
cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
