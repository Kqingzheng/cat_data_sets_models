from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


from tensorboard.plugins.hparams import keras
from tensorflow import optimizers
import tensorflow as tf
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
from keras.applications import ResNet101V2

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from PIL import Image
import matplotlib.pyplot as plt
import keras.backend as K
from tensorflow.python.keras.optimizer_v2.nadam import Nadam


# def model():
#
#     conv_base = ResNet101V2(weights='imagenet')
#     conv_base.trainable = False
#     model = models.Sequential()
#     model.add(conv_base)
#     # model.add(GlobalAveragePooling2D())
#     # model.add(Dense(512, activation='relu'))
#     # # model.add(Flatten())
#     # model.add(Dense(12, activation='softmax'))
#     # conv_base = ResNet101V2(weights='imagenet', input_shape=(150, 150, 3), include_top=False)
#     # conv_base.trainable = False
#     # model = models.Sequential()
#     # model.add(conv_base)
#     # model.add(GlobalAveragePooling2D())
#     # model.add(Dense(512, activation='relu'))
#     # model.add(Dense(512, activation='relu'))
#     # model.add(Dense(12, activation='softmax'))
#
#     return model

# model=model()
# model.summary()

# def predict_cat(url):
#     path = "./whichcat.jpeg"
#     img = image.load_img(path, target_size=(150, 150))
#     x = image.img_to_array(img) / 255.
#     x = x.reshape((1,) + x.shape)
#     model = load_model('model1.hdf5')
#     xx = model.predict_classes(x)
#     print(xx[0])
conv_base = ResNet101V2(weights='imagenet', input_shape=(150, 150, 3), include_top=False)
i=0
for layer in conv_base.layers:
    i=i+1
print(i)



