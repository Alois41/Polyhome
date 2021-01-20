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


# determine if a candidate face is a match for a known face
def is_match(_known_embedding, _candidate_embedding, thresh=0.0001):
    # calculate distance between embeddings
    score = cosine(_known_embedding, _candidate_embedding)
    return score


if __name__ == "__main__":
    # pre-trained model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # # Known User
    # temp = []
    # dir = r"./data/train/12220b50edfc438abbd6b9de78be6e09/"
    # for path in os.listdir(dir):
    #     im = cv2.imread(dir+path)
    #     im = cv2.normalize(im.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    #     temp.append(np.asarray(cv2.resize(im, (224, 224)), 'float32'))
    # known_embedding_alois = model.predict(preprocess_input(temp, version=2))
    #
    # temp = []
    # dir = r"./data/train/f43e71c31d724af3a7f16258924a095e/"
    # for path in os.listdir(dir):
    #     im = cv2.imread(dir+path)
    #     im = cv2.normalize(im.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    #     temp.append(np.asarray(cv2.resize(im, (224, 224)), 'float32'))
    # known_embedding_alexis = model.predict(preprocess_input(temp, version=2))

    f_getter = FaceGetter()
    plt.ion()
    embed = []
    dirs = [r"./data/train/88f1c6433ea84e7f8acd6ec59dd3738b/",
            r'./data/train/1fd8914c487046d1be2a10ad019b063f/',
            r'./data/train/752fc44264224bec9039633fd5e0ac6c/',
            r'./data/train/461a3ec07b2243ea95b956b784e437ac/']

    gens = ['alois', 'alexis', 'tanguy', 'morgan']
    i = 0
    for dir in dirs:
        for path in os.listdir(dir):
            im = cv2.imread(dir + path)
            im = cv2.normalize(im.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
            im = np.asarray(cv2.resize(im, (224, 224)), 'float32')
            embed.append({'data': model.predict(preprocess_input([im], version=2)),
                          'im': im, 'label': gens[i]})
        i += 1

    hist = [[np.NAN] * 20, [np.NAN] * 20, [np.NAN] * 20, [np.NAN] * 20] * 4
    while True:
        face = None

        while face is None:
            face = f_getter.get_face()

        try:
            image = cv2.resize(face, (224, 224))
            image = cv2.normalize(image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
            unknown_embedding = model.predict(preprocess_input([image], version=2), use_multiprocessing=True)

            scores = {}
            bests = {}
            im_bests = {}
            for item in embed:
                score = is_match(item['data'], unknown_embedding)
                if item['label'] in scores and item['label'] in bests:
                    scores[item['label']] += score
                    last = bests[item['label']]
                    if last > score:
                        bests[item['label']] = score
                        im_bests[item['label']] = item['im']
                else:
                    scores[item['label']] = score
                    bests[item['label']] = score
                    im_bests[item['label']] = item['im']

            hist[0].pop(0)
            hist[1].pop(0)
            hist[2].pop(0)
            hist[3].pop(0)
            hist[0].append(scores[gens[0]])
            hist[1].append(scores[gens[1]])
            hist[2].append(scores[gens[2]])
            hist[3].append(scores[gens[3]])

            plt.subplot(221)
            plt.title("History")
            plt.plot(hist[0], color='blue', label='Aloïs', marker='o')
            plt.plot(hist[1], color='red', label='Alexis', marker='o')
            plt.plot(hist[2], color='black', label='Tanguy', marker='o')
            plt.plot(hist[3], color='green', label='Morgan', marker='o')
            plt.legend()
            plt.subplot(223)
            plt.title("Sample")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.subplot(224)
            plt.title("Best match")
            plt.imshow(cv2.cvtColor(im_bests[gens[np.argmax([hist[0][-1], hist[1][-1], hist[2][-1], hist[3][-1]])]], cv2.COLOR_BGR2RGB))
            plt.pause(0.0001)
            plt.clf()

        except Exception as e:
            print(str(e))
            continue
