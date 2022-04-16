import cv2 as cv
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
cap = cv.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        is_okay, image = cap.read()
        image = cv.flip(image, 1)
        if not is_okay:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.detections:
            for face_no, face in enumerate(results.detections):
                # Draw the face bounding box and key points on the copy of the input image.
                mp_drawing.draw_detection(image=image, detection=face,
                                          # # keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                          # #                                              thickness=-1,
                                          # #                                              circle_radius=3),
                                          # bbox_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                          #                                          thickness=2))
                                          )

                # Retrieve the bounding box of the face.
                face_bbox = face.location_data.relative_bounding_box

                # Retrieve the required bounding box coordinates and scale them according to the size of original
                # input image.
                x1 = int(face_bbox.xmin * image_width)
                y1 = int(face_bbox.ymin * image_height)

                # Draw a filled rectangle near the bounding box of the face.
                # We are doing it to change the background of the confidence score to make it easily visible
                # cv.rectangle(image, pt1=(x1, y1 - image_width // 20), pt2=(x1 + image_width // 16, y1),
                #              color=(0, 255, 0), thickness=1)

                # Write the confidence score of the face near the bounding box and on the filled rectangle.
                cv.putText(image, text=f"{(round(face.score[0], 1) * 100)}%", org=(x1, y1 - 25),
                           fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=image_width // 700, color=(255, 255, 255),
                           thickness=2)

        cv.imshow('MediaPipe Face Detection', image)
        if cv.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
