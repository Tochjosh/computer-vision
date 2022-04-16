import mediapipe as mp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


class Media:
    mp_holistic = mp.s

    def detect(self):
        cam = cv.VideoCapture(0)

        while cam.isOpened():
            is_okay, frame = cam.read()
            cv.imshow('cam feed', cv.flip(frame, 1))

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cam.release()
        cv.destroyAllWindows()



