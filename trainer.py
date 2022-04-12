import cv2 as cv
import numpy as np
import os
from PIL import Image


class Trainer:

    def __init__(self):
        self.base_path = 'dataset'
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    def get_image_and_label(self, path):
        directory = os.path.join(self.base_path, path)
        image_paths = [os.path.join(directory, image) for image in os.listdir(directory)]

        face_samples = []
        ids = []
        for image_path in image_paths:
            image = Image.open(image_path)
            grey = image.convert('L')  # converts image to greyscale with Pillow
            image_numpy = np.array(grey, 'uint8')
            image_id = int(os.path.split(image_path)[1])
            faces = self.detector.detectMultiScale(image_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(image_numpy[y:y + h, x:x + w])
                ids.append(image_id)

        return face_samples, ids

    def train_face(self, path):

        print('Training faces...')
        faces, ids = self.get_image_and_label(path)
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write('trainer/trainer.yml')
        self.recognizer.save()
