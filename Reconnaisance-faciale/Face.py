import ctypes

from mtcnn.mtcnn import MTCNN
import cv2

import numpy as np
import time


class FaceGetter:
    def __init__(self):
        self.detector = MTCNN()
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.margins = 100
        self.w = 224
        self.h = 224

    def get_face(self):
        ret, frame = self.cam.read()
        frame = cv2.flip(frame, 1)
        result = (self.detector.detect_faces(frame))
        if len(result) > 0 and ret:
            x1, y1, width, height = result[0]['box']
            x2, y2 = x1 + width, y1 + height
            frame = frame[y1:y2, x1:x2]
            return frame
        return None



