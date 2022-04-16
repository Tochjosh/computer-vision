import cv2 as cv
import numpy as np
import face_recognition
import os
import pickle
from PIL import Image


class FacialRecognition:

    def collect_dataset(self):

        name = input('Enter your name: ')
        if len(os.listdir('dataset')) == 0:
            id = 1
        else:
            id = len(os.listdir('dataset')) + 1

        cam = cv.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        count = 1
        print(f"capturing {name}'s facial dataset...")

        while cam.isOpened():
            directory = f'dataset/{name}.{id}'
            filename = f"{directory}/{str(count)}.jpg"
            if f"{name}.{id}" not in os.listdir('dataset'):
                os.mkdir(directory)
            is_working, frame = cam.read()
            if not is_working:
                continue
            grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imwrite(filename, grey)
            count += 1
            cv.imshow(f"Collecting {name}'s face", frame)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break
            if count >= 21:
                break
        cam.release()
        cv.destroyAllWindows()
        print('Process complete, and images saved to a folder')

    def train(self):
        recognizer = cv.face.LBPHFaceRecognizer_create()
        detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

        people = [os.path.join('dataset', person) for person in os.listdir('dataset')]

        # files = [os.path.join(directories, f) for f in directories]
        faces = []
        labels = []

        for person in people:
            file_paths = os.listdir(person)
            print(file_paths)
            for file_path in file_paths:
                file = os.path.join(person, file_path)
                print(file)
                img_numpy = np.array(Image.open(file), 'uint8')
                face_id = int(os.path.split(file)[0].split(".")[1])
                detected_faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in detected_faces:
                    faces.append(img_numpy[y:y + h, x:x + w])
                    labels.append(face_id)

        recognizer.train(faces, np.array(labels))
        recognizer.write('trainer/trainer.yml')
        print(f"{len(np.unique(labels))} faces trained. Exiting Program")


    def


if __name__ == '__main__':
    fr = FacialRecognition()
    # fr.collect_dataset()
    fr.train()
