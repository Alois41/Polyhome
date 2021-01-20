from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras.applications.resnet50 as resnet
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Dense, \
    BatchNormalization
from multiprocessing import Array, freeze_support, Value
from keras.models import Model
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA


# determine if a candidate face is a match for a known face
def is_match(_known_embedding, _candidate_embedding, thresh=0.0001):
    # calculate distance between embeddings
    score = cosine(_known_embedding, _candidate_embedding)
    return score


if __name__ == "__main__":
    # pre-trained model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    embed = []
    labels = []
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
        i += 1

    pca = PCA(n_components=2)
    X = np.array(embed)
    X = np.reshape(X, (np.shape(embed)[0], 2048))
    X = pca.fit_transform(X)

    i = 0
    for pt in X:
        plt.scatter(pt[0], pt[1], color=labels[i])
        i += 1

    plt.show()
