import numpy as np
import tensorflow as tf
from keras.api.models import Model, load_model
from keras.api.preprocessing import image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import pyplot as plt


# 加载模型
sequential_model = load_model('./TuberculosisImageClassifier.keras')

# 打印模型结构，确认 'conv2d_2' 层存在
sequential_model.summary()


# 准备输入图像
img_path = "/Users/siou/Downloads/code/TuberculosisImageClassifier/TB_Chest_Radiography_Database/Normal/Normal-5.png"
img = image.load_img(img_path, target_size=(224, 224, 3))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
img_array = img_array / 255.0  # 归一化处理


# 选择目标类别的评分函数
predicted_class = 1  # 假设你想查看类别 0 (normal) 的 Grad-CAM
score = CategoricalScore([predicted_class])

# 创建 Grad-CAM 实例
gradcam = Gradcam(sequential_model)

# 尝试运行模型推理
conv_output = sequential_model.predict(img_array)
print("卷积层输出形状:", conv_output)

# 生成热力图
heatmap = gradcam(
    score,
    img_array,
    penultimate_layer=sequential_model.get_layer('conv2d_2'))

# 可视化热力图
plt.imshow(heatmap[0], cmap='jet', alpha=0.2)
plt.show()
