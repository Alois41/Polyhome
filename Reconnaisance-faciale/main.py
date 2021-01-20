import ctypes

import cv2
from Face import FaceGetter
import numpy as np
import sys
import os
import time
import uuid


if __name__ == "__main__":
    f_getter = FaceGetter()
    folder = uuid.uuid4().hex
    os.makedirs("./data/train/%s" % folder)

    face = None

    for i in range(20):
        face = None
        while face is None:
            face = f_getter.get_face()

        cv2.imwrite("./data/train/%s" % folder + "/image_%i.jpg" % i, face)
        print("image saved %0.2f%%" % (i/20.0))

        cv2.imshow("retour", face)
        key = cv2.waitKey(1)
        time.sleep(1)
        if key == 27:
            continue

    exit(0)


