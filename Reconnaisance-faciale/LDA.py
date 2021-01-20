from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras.applications.resnet50 as resnet
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Dense, \
    BatchNormalization
from multiprocessing import Array, freeze_support, Value
from keras.models import Model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Face import FaceGetter
import ctypes
import time
import cv2
import numpy as np
import os
import tensorflow as tf
import random
import keras
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# determine if a candidate face is a match for a known face
def is_match(_known_embedding, _candidate_embedding, thresh=0.0001):
    # calculate distance between embeddings
    score = cosine(_known_embedding, _candidate_embedding)
    return score


if __name__ == "__main__":
    # pre-trained model
    f_getter = FaceGetter()

    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    embed = []
    labels = []
    Y = []
    dirs = [r"./data/face_images/alois/",
            r'./data/face_images/alexis/',
            r'./data/face_images/tanguy/',
            r'./data/face_images/morgan/']

    gens = ['alois', 'alexis', 'tanguy', 'morgan']
    colors = ['red', 'blue', 'green', 'black']
    i = 0
    for dir in dirs:
        for path in os.listdir(dir):
            im = cv2.imread(dir + path)
            im = cv2.normalize(im.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
            im = np.asarray(cv2.resize(im, (224, 224)), 'float32')
            embed.append(model.predict(preprocess_input([im], version=2)))
            labels.append(colors[i])
            Y.append(gens[i])
        i += 1

    clf = LDA()
    X = np.array(embed)
    X = np.reshape(X, (np.shape(embed)[0], 2048))
    X = clf.fit_transform(X, Y)

    plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)

    while True:

        face = None

        while face is None:
            face = f_getter.get_face()

        try:
            image = cv2.resize(face, (224, 224))
        except Exception as e:
            continue

        image = cv2.normalize(image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        unknown_embedding = model.predict(preprocess_input([image], version=2), use_multiprocessing=True)
        new_point = clf.transform(unknown_embedding)[0]
        print(clf.predict(unknown_embedding))

        ax.clear()
        ax.scatter(*new_point, marker='d')
        i = 0
        for pt in X:
            ax.scatter(*pt, color=labels[i])
            i += 1
        plt.title("Face embeddings with linear discriminant analysis")


        plt.show()
        plt.pause(0.1)





