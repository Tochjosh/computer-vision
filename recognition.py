import cv2 as cv
import numpy as np
import face_recognition
import os
import pickle


class FacialRecognition():

    def collect_dataset(self):

        name = input('Enter your name: ')
        path = 'dataset'
        if len(os.listdir(path)) > 0:
            person_id = int(os.listdir(path)[-1]) + 1
        else:
            person_id = 1
        cam = cv.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        count = 1

        while cam.isOpened():
            directory = f'dataset/{name}'
            filename = f"{directory}/{name}.{str(count)}.jpg"
            if name not in os.listdir('dataset'):
                os.mkdir(directory)
            is_working, frame = cam.read()
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


if __name__ == '__main__':
    fr = FacialRecognition()
    fr.collect_dataset()
