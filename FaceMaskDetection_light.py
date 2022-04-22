import cv2
import dlib
from imutils import face_utils
import numpy as np
from keras.models import load_model

tip = 'no mask'
tip_no_face = 'no face'
tip_fake_face = 'fake face'
'''tip_mask = 'mask'
tip_no_mask = 'no mask'''
font = cv2.FONT_HERSHEY_SIMPLEX

time_count = 1
frame_rate = 70
EAR_thresh = 0.3
eye_low_consecutive_frames = 3
low_frame_count = 0
blink_sum = 0

face_detector = dlib.get_frontal_face_detector()
keypoint_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

model = load_model('mask_all.h5')
dataset = []


def EAR_calculator(eye):
    p2p6 = np.linalg.norm(eye[1] - eye[5])
    p3p5 = np.linalg.norm(eye[2] - eye[4])
    p1p4 = np.linalg.norm(eye[0] - eye[3])
    EAR = (p2p6 + p3p5) / (2.0 * p1p4)
    return EAR


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if ret is None:
        break
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_img, 0)
    if len(faces) == 0:
        cv2.putText(img, tip_no_face, (70, 50), font, 2, (255, 0, 0), 2, 3)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        shape = keypoint_predictor(gray_img, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]
        left_EAR = EAR_calculator(left_eye)
        right_EAR = EAR_calculator(right_eye)

        EAR = (left_EAR + right_EAR) / 2.0

        left_hull = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)

        cv2.drawContours(img, [left_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [right_hull], -1, (0, 255, 0), 1)

        if EAR < 0.3:
            low_frame_count += 1

        if low_frame_count >= 3:
            blink_sum += 1
            low_frame_count = 0

        if blink_sum >= 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, "Blinks: {}".format(blink_sum), (x1 + 10, y1 - 35), font, 1, (0, 0, 255), 2)
            cv2.putText(img, tip, (x1 + 10, y1 - 10), font, 1, (0, 0, 255), 2)

            if ret:
                if time_count % frame_rate == 0:
                    cut_img = img[y1:y2, x1:x2]
                    resize_img = cv2.resize(cut_img, (128, 128), cv2.INTER_AREA)
                    dataset.append(resize_img)
                    data_test = np.array(dataset)
                    predict = model.predict(data_test)
                    result_class = np.argmax(predict, axis=1)
                    if result_class[len(result_class) - 1] == 1:
                        tip = 'mask'
                        # cv2.putText(img, tip_mask, (x1 + 10, y1 - 10), font, 1, (0, 0, 255), 2, 3)
                    else:
                        tip = 'no mask'
                        # cv2.putText(img, tip_no_mask, (x1 + 10, y1 - 10), font, 1, (0, 0, 255), 2, 3)
                time_count += 1

        else:
            cv2.putText(img, tip_fake_face, (x1 + 10, y1 - 10), font, 1, (0, 0, 255), 2, 3)

        if blink_sum >= 3:
            blink_sum = 0

    cv2.imshow('face mask detector', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
