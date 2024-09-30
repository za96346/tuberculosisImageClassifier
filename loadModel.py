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
img_path = '/Users/siou/Downloads/code/TuberculosisImageClassifier/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-677.png'  # 替换为你的测试图像路径
img = image.load_img(img_path, target_size=(512, 512))
img = np.asarray(img)
plt.imshow(img)

img = np.expand_dims(img, axis=0)





plt.show()