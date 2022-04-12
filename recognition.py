import cv2 as cv
import numpy as np
import face_recognition
import os
import pickle


class FacialRecognition:

    def collect_dataset(self):

        name = input('Enter your name: ')
        path = 'dataset'
        if len(os.listdir(path)) > 0:
            person_id = int(os.listdir(path)[-1]) + 1
        else:
            person_id = 1
        capture = cv.VideoCapture(0)
        capture.set(3, 640)  # set video width
        capture.set(4, 480)  # set video height
        count = 1

        while capture.isOpened():
            is_working, frame = capture.read()
            # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            file_dir = os.path.join(path, str(person_id))
            filename = f"{file_dir}/{name}.{str(count)}.jpg"
            print(filename)
            cv.imwrite(filename, frame)
            count += 1
            cv.imshow(f"Collecting {name}'s face", frame)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break
            # if count >= 21:
            #     break
        capture.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    fr = FacialRecognition()
    fr.collect_dataset()
