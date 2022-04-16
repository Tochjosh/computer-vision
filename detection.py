import cv2 as cv


class FaceDetection:

    def detect_face(self):

        cam = cv.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        net = cv.dnn.readNetFromCaffe(prototxt="deploy.prototxt",
                                      caffeModel="res10_300x300_ssd_iter_140000.caffemodel")

        while cam.isOpened():
            is_okay, frame = cam.read()
            if not is_okay:
                continue
                
            image_height, image_width, _ = frame.shape
            preprocessed_image = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                                      mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
            output_image = frame.copy()
            net.setInput(preprocessed_image)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    bounding_box = detections[0, 0, i, 3:7]

                    x1 = int(bounding_box[0] * image_width)
                    y1 = int(bounding_box[1] * image_height)
                    x2 = int(bounding_box[2] * image_width)
                    y2 = int(bounding_box[3] * image_height)

                    cv.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),
                                 thickness=2)

                    cv.putText(output_image, text=f"{str(round(confidence, 1) * 100)}%", org=(x1, y1 - 25),
                               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.45,
                               color=(255, 255, 255), thickness=2)

                    cv.imshow("Facial Detection", output_image)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cam.release()
        cv.destroyAllWindows()


if __name__ == '__main__':

    face_detection = FaceDetection()
    face_detection.detect_face()
