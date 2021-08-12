from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet101V2
from tensorboard.plugins.hparams import keras
from tensorflow import optimizers
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model

import matplotlib.pyplot as plt



def model():
    conv_base = ResNet101V2(weights='imagenet', input_shape=(150, 150, 3), include_top=False)

    for layer in conv_base.layers[0:140]:
        layer.trainable = False
    for layer in conv_base.layers[140:0]:
        layer.trainable = True


    model = models.Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(12, activation='softmax'))


    return model

#  训练数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   #  表示图像随机旋转的角度范围
                                   rotation_range=40,
                                   #  图像在水平方向上平移的范围
                                   width_shift_range=0.2,
                                   #  图像在垂直方向上平移的范围
                                   height_shift_range=0.2,
                                   #  随机错切变换的角度
                                   shear_range=0.2,
                                   #  图像随机缩放的范围
                                   zoom_range=0.2,
                                   #  随机将一半图像水平翻转
                                   horizontal_flip=True)
#  验证数据不能增强
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = r"./target_K_train_data_sets/"
validation_dir = r"./target_K_validation_data_sets/"

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    #  将所有图像的大小调整为227*227
                                                    target_size=(150, 150),
                                                    #  批量大小
                                                    batch_size=16,
                                                    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=16,
                                                              class_mode='categorical')

model = model()
#  用于配置训练模型（优化器、目标函数、模型评估标准）

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#  查看各个层的信息
model.summary()
#  回调函数，在每个训练期之后保存模型和设置早停
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, verbose=1), # 当两次迭代损失未改善，Keras停止训练
    ModelCheckpoint('modelResNet101V2inK.hdf5',  # 保存模型的路径
                    monitor='loss',  # 被监测的数据
                    verbose=1,  # 日志显示模式:0=>安静模式,1=>进度条,2=>每轮一行
                    save_best_only=True),  # 若为True,最佳模型就不会被覆盖,
]


#  用history接收返回值用于画loss/acc曲线
history = model.fit(train_generator,
                    steps_per_epoch=105,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=30
                  ) #  validation_data=validation_generator,validation_steps=30



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()