import numpy as np
import tensorflow as tf
from keras.api.models import load_model, Model
from keras.api.preprocessing import image
import matplotlib.pyplot as plt

# 加载训练好的模型
model = load_model('my_vgg16_model.h5')

# 打印模型摘要
model.summary()

# 提取 VGG16 子模型（去掉全连接层部分）
vgg16_model = model.get_layer('vgg16')
vgg16_model.summary()

# 获取 VGG16 最后一个卷积层的输出
last_conv_layer = vgg16_model.get_layer('block5_conv3')

# 建立一个模型，输出最后一个卷积层的激活值
intermediate_model = Model(inputs=vgg16_model.input, outputs=last_conv_layer.output)

# 加载和预处理图片
img_path = '/Users/siou/Downloads/code/TuberculosisImageClassifier/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-677.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

conv_output = intermediate_model.predict(img_array)

classifier_weights = model.get_layer('dense_2').get_weights()[0]

predicted_class = np.argmax(model.predict(img_array), axis=-1)[0]

class_weights = classifier_weights[:, predicted_class]

cam_output = np.dot(conv_output[0], class_weights)

cam_output = np.maximum(cam_output, 0)
cam_output = cam_output / cam_output.max()

cam_output = tf.image.resize(cam_output[..., np.newaxis], (224, 224))

plt.imshow(tf.keras.preprocessing.image.array_to_img(img_array[0]))
plt.imshow(cam_output, cmap='jet', alpha=0.5)
plt.colorbar()
plt.show()