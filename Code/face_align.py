import numpy as np
import cv2 as cv
import dlib
import cv2
import argparse
import os
from collections import OrderedDict
# referred from publicly available site

class FaceAligner(object):

    def __init__(self, predictor, desiredLeftEye = (0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        self.FACIAL_LANDMARKS_IDX = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 35)),
            ("jaw", (0, 17))
        ])

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, rect):
        shape = self.predictor(image, rect)
        shape = shape_to_np(shape)

        (lStart, lEnd) = self.FACIAL_LANDMARKS_IDX["left_eye"]
        (rStart, rEnd) = self.FACIAL_LANDMARKS_IDX["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        dist = np.sqrt((dX**2) + (dY**2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth

        scale = desiredDist/ dist
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0])//2, (leftEyeCenter[1] + rightEyeCenter[1])//2)

        M = cv.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0,2] += (tX - eyesCenter[0])
        M[1,2] += (tY - eyesCenter[1])

        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv.warpAffine(image, M, (w,h), flags=cv.INTER_CUBIC)

        return output


# rectangle to bounding box
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y ,w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def get_facial_lanmarks(image, path):
    path_to_pred = os.path.join(path, "code", "shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_pred)

    rects = detector(image)

    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = shape_to_np(shape)

        return shape,

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    path1 = os.path.join(path, 'data', 'lfw', 'Aaron_Peirsol', )

    path_to_pred = os.path.join(path, "code", "shape_predictor_68_face_landmarks.dat")



    image = cv.imread(os.path.join(path1, 'Aaron_Peirsol_0004.jpg'), 0)
    image = cv.resize(image,(300,300))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_pred)


    ## Face Alignment------------------------------
    # align_img = FaceAligner(predictor)
    # rects = detector(image,2)
    # for rect in rects:
    #     cv.imshow('before', image)
    #     image = align_img.align(image,rect)
    #     break
    ## Landmark Detection--------------------------
    # rects = detector(image)
    #
    # print(type(rects))
    #
    # for (i, rect) in enumerate(rects):
    #
    #     shape = predictor(image,rect)
    #     shape = shape_to_np(shape)
    #
    #     (x,y,w,h) = rect_to_bb(rect)
    #     cv2.rectangle(image, (x,y),(x+w, y+h),(0.255,0), 2)
    #
    #     for (x,y) in shape:
    #         cv2.circle(image, (x,y), 1 , (0,0,255), -1)

    cv.imshow("out", image)
    cv.waitKey(0)

