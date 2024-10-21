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
