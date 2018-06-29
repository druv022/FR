import cv2 as cv
import numpy as np
import os
import pickle
import PIL
import process_lwf
import process_pipa
import process_adience


def detect_face(image):
    face_cascade = cv.CascadeClassifier(os.path.join('C:', 'Users\Druv\AppData\Local\conda\conda\envs\OpenCV\Lib'
                                                           '\site-packages\cv2\data\haarcascade_frontalface_default.xml'))

    faces = face_cascade.detectMultiScale(image, 1.2, 25)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]

    return image[y:y + w, x:x + h], faces[0]

def dump_face_into_folder(face_dict, dataset_name, folder_path):
    for identity in face_dict:
        faces = face_dict[identity]
        dir = os.path.join(folder_path, dataset_name, str(identity))
        if not os.path.exists(dir):
            os.makedirs(dir)

        count = 1
        for face in faces:

            if not isinstance(face, str):
                cv.imwrite(os.path.join(dir, str(count) + '.jpg'), face)
            else:
                try:
                    image = cv.imread(face, 0)
                    if image is None:
                        continue
                    face, location = detect_face(image)
                    if face is None:
                        continue
                    cv.imwrite(os.path.join(dir, str(count) + '.jpg'), face)
                except:
                    continue

            count += 1

def stack_horizontally(path):
    list_dir = os.listdir(path)
    for dir in list_dir:
        images_path = os.path.join(path, dir)
        images = os.listdir(images_path)

        face_list = [PIL.Image.open(os.path.join(images_path, i)) for i in images if
                     cv.imread(os.path.join(images_path, i)) is not None]

        min_shape = sorted([(np.sum(i.size), i.size) for i in face_list])
        if len(min_shape) > 0:
            min_shape = min_shape[0][1]
            imgs_comb = np.hstack((cv.resize(np.asarray(i), min_shape) for i in face_list))

            imgs_comb = PIL.Image.fromarray(imgs_comb)
            imgs_comb.save(os.path.join(images_path, 'faces_' + str(dir) + '.jpg'))


def prepare_sets(path,dataset_path,dataset_name,min_train_img=5, tr_percnt=0.5):
    list_dir = os.listdir(dataset_path)

    training_set = {}
    test_set = {}

    for identity in list_dir:
        list_files = os.listdir(os.path.join(dataset_path, identity))
        number_faces = len(list_files)

        no_train_img = int(tr_percnt * number_faces)
        if number_faces < 2 * min_train_img:
            continue
        else:
            index = range(number_faces)
            index = np.random.permutation(index)
            for i in index:
                if i in index[0:no_train_img]:
                    if identity not in training_set.keys():
                        training_set[identity] = []
                    value = training_set[identity]
                    value.append((os.path.join(dataset_path, identity, list_files[i])))
                    training_set[identity] = value
                else:
                    value = test_set
                    if identity not in test_set.keys():
                        test_set[identity] = []
                    value = test_set[identity]
                    value.append(os.path.join(dataset_path, identity, list_files[i]))
                    test_set[identity] = value

        print(identity)

    with open(os.path.join(path,'DUMP',dataset_name, 'train_test.pkl'), 'wb+') as f:
        pickle.dump([training_set, test_set], f)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'faces')

    # For PIPA
    # face_dict, id_dict = process_pipa.get_pipa_faces()
    # For LWF
    # face_dict = process_lwf.get_lwf_faces()
    # For adience aligned
    # face_dict = process_adience.get_adience_aligned_faces()

    # change the dataset name accordingly before running
    # 'LWF', 'PIPA'
    # dataset = 'LWF'
    # dump_face_into_folder(face_dict, dataset, path)

    # dataset_path = os.path.join(path, 'LWF')
    # prepare_sets(dataset_path, tr_percnt=0.8)
