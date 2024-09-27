import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# 載入 VGG-19 模型，設定 include_top=False 以移除預設的分類層
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 鎖定 VGG19 預訓練的層，這樣它們不會在訓練中被更新
vgg19_base.trainable = False

# 建立自定義的分類層
model = models.Sequential()
model.add(vgg19_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 假設二元分類，根據需要調整為多分類

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

# 儲存模型
model.save('vgg19_finetuned_model.h5')


# 解鎖 VGG-19 的部分層
for layer in vgg19_base.layers[-4:]:
    layer.trainable = True

# 重新編譯模型，使用更小的學習率
model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 繼續訓練
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)
