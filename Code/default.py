import cv2 as cv
import os
import numpy as np
import pickle
import data
from sklearn.metrics import classification_report


def detect_face(img):
    face_cascade = cv.CascadeClassifier(os.path.join('C:', 'Users\Druv\AppData\Local\conda\conda\envs\OpenCV\Lib'
                                                           '\site-packages\cv2\data\haarcascade_frontalface_default.xml'))
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]

    return img[y:y + w, x:x + h], faces[0]


def prepare_data(dataset, no_id=2, no_examples=50, min_sample=False):
    faces = []
    labels = []
    min_count = 0

    if min_sample and no_id != 0:
        min_count = min([len(dataset[id]) for i, id in enumerate(dataset) if i <= no_id])
        print("Minimum number of image used: ", min_count)

    for index, identity in enumerate(dataset):
        count = 0
        for i, face_p in enumerate(dataset[identity]):
            image_1 = cv.imread(face_p, 0)
            image_1 = cv.resize(image_1, (300, 300))
            if image_1 is not None:
                faces.append(image_1)
                count += 1
                labels.append(int(identity))

            if no_examples != 0 and (count >= no_examples or (min_count and count >= min_count)):
                break

        if no_id != 0and index >= no_id:
            break

    return faces, labels


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)


def predict(test_img, face_img=True):
    img = test_img.copy()
    rect = None
    face = test_img
    if not face_img:
        face, rect = detect_face(img)

    if face is not None:
        face = cv.resize(face, (300, 300))
        label = face_recognizer.predict(face)
    else:
        label = face_recognizer.predict(img)
    label_text = str(label)

    if rect is not None:
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1] - 5)
    # print(label)

    return label


def predict_set(faces):
    labels = []
    labels_prob = []
    for face in faces:
        pred = predict(face, face_img=True)
        labels.append(pred[0])
        labels_prob.append(pred[1])

    return labels, labels_prob


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    train_set, test_set = data.get_dataset('LFW', path,min_img=5)

    num_id = 0
    train_faces, train_labels = prepare_data(train_set, no_id=num_id, no_examples=0, min_sample=False)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    # face_recognizer = cv.face.EigenFaceRecognizer_create()
    # face_recognizer = cv.face.FisherFaceRecognizer_create()

    # print(train_labels)
    # [print(i,train_faces[i].size) for i,face in enumerate(train_faces)]
    face_recognizer.train(train_faces, np.array(train_labels))

    test_faces, test_labels = prepare_data(test_set, no_id=num_id, no_examples=5)

    # [print(i, test_faces[i].size) for i, face in enumerate(test_faces)]
    pred_labels, pred_prob = predict_set(test_faces)
    # [print(test_labels[i], pred_labels[i]) for i in range(len(test_labels))]
    #
    print(classification_report(test_labels, pred_labels))


    #
    # test_image1 = cv.imread(os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '1', '4_.jpg'),0)
    # test_image2 = cv.imread(os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '100', '1_.jpg'),0)
    # test_image3 = cv.imread(os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '1008', '4_.jpg'), 0)
    # test_image4 = cv.imread(os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '1011', '1_.jpg'), 0)
    # test_image5 = cv.imread(os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '1051', '4_.jpg'), 0)
    #
    # predict_img1 = predict(test_image1)
    # predict_img2 = predict(test_image2)
    # predict_img3 = predict(test_image3)
    # predict_img4 = predict(test_image4)
    # predict_img5 = predict(test_image5)
    #
    # cv.imshow('1', predict_img1)
    # cv.waitKey(0)
    # cv.imshow('100', predict_img2)
    # cv.waitKey(0)
    # cv.imshow('1008', predict_img3)
    # cv.waitKey(0)
    # cv.imshow('1011', predict_img4)
    # cv.waitKey(0)
    # cv.imshow('1051', predict_img5)
    # cv.waitKey(0)
    # cv.imshow('100', predict_img2)
    # cv.waitKey(0)
