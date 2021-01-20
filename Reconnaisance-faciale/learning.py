from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras.applications.resnet50 as resnet
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Dense, \
    BatchNormalization
from multiprocessing import Array, freeze_support, Value
from keras.models import Model
from Face import FaceGetter
import ctypes
import time
import cv2
import numpy as np
import os
import tensorflow as tf
import random
import keras
from keras_vggface.vggface import VGGFace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gens = {0: "Alois", 1: "Morgan"}

if __name__ == "__main__":
    from keras.engine import Model
    from keras.layers import Input

    # Convolution Features
    vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')

    model = Sequential()

    model.add(vgg_features)
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    x_train_datagen = []
    for path in os.listdir(r"./data/train/"):
        for i in range(10):
            r = random.choice([x for x in os.listdir(r"./data/train/" + path)
                               if os.path.isfile(os.path.join(r"./data/train/" + path, x))])
            print(r"./data/train/" + path + r)
            x_train_datagen.append(cv2.imread(r"./data/train/%s/%s" % (path, r)))

    print(model.summary())

    # datagen = ImageDataGenerator(featurewise_center=True,
    #                              featurewise_std_normalization=True,
    #                              rotation_range=5, zoom_range=.1,
    #                              width_shift_range=0.2,
    #                              height_shift_range=0.2)
    # datagen.fit(x_train_datagen)
    # del x_train_datagen
    #
    # train_generator = datagen.flow_from_directory(r"./data/train/", batch_size=10, target_size=(224, 224),
    #                                               color_mode="rgb",
    #                                               class_mode="categorical", shuffle=True)
    # validation_generator = datagen.flow_from_directory(r"./data/validation/", batch_size=5, target_size=(224, 224),
    #                                                    color_mode="rgb",
    #                                                    class_mode="categorical", shuffle=True)
    #
    # test_generator = datagen.flow_from_directory(r"./data/test/", batch_size=10, target_size=(224, 224),
    #                                              color_mode="rgb",
    #                                              class_mode="categorical", shuffle=True)
    #
    # model.fit_generator(generator=train_generator, steps_per_epoch=100, epochs=5,
    #                     validation_data=validation_generator, validation_steps=100)
    #
    # print(model.evaluate_generator(test_generator))

    a = Array(ctypes.c_float, 224 * 224 * 3)
    isFaceHere = Value('i', False)
    f_getter = FaceGetter(a, isFaceHere)
    f_getter.daemon = True
    f_getter.start()
    test_data = []
    for i in range(100):
        while not isFaceHere.value:
            time.sleep(.1)

        with a.get_lock():
            image = np.frombuffer(a.get_obj(), dtype=np.float32)
            image = np.reshape(image, (224, 224, 3))

        cv2.imshow('test', image)
        image = cv2.resize(image, (224, 224))
        predict = model.predict_classes(np.reshape(image, (1, *image.shape)))
        test_data.append([predict])

        key = cv2.waitKey(10)
        if key == 27:
            break

    print(np.mean(test_data))
