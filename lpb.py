import cv2 as cv

cam = cv.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
lbp_face_cascade = cv.CascadeClassifier('lbpcascade_frontalface.xml')
# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize individual sampling face count
# count = 0
while cam.isOpened():
    ret, img = cam.read()
    img = cv.flip(img, 1)  # flip video image vertically
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = lbp_face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # count += 1
        # Save the captured image into the datasets folder
        # cv.imwrite("dataset/User." + str(face_id) + '.' +
        #            str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv.imshow('image', img)
    # k = cv.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    # elif count >= 30:  # Take 30 face sample and stop video
    #     break
# Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv.destroyAllWindows()
