import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import pickle


def read_index(index_folder):
    '''
    This read the index file and stores the information id a dictionary.
    :param index_folder: path to index file
    :return: dictionary of ID
    '''
    id_dict = {}
    with open(index_folder, 'r') as f:
        for line in f:
            data = f.readline().split()

            # if Person identity not in the dict, initialize the list
            if data[6] not in id_dict.keys():
                id_dict[data[6]] = []
            else:
                value = id_dict[data[6]]
                # value format: tuple(tuple(photoset_id, photo_id),tuple(xmin,ymin,width,height))
                value.append(((data[0], data[1], data[7]), (data[2], data[3], data[4], data[5])))
                id_dict[data[6]] = value

    return id_dict


def get_image(files, path_img, photoset_id, photo_id):
    '''
    Return the image for given information
    :param files: path of the images folder. Used for debugging
    :param path_img: path to the image
    :param photoset_id: Photo album identifier
    :param photo_id: photo_id or photo identifier
    :return:
    '''
    files = os.listdir(path_img)

    filename = str(photoset_id) + '_' + str(photo_id) + '.jpg'
    filepath = os.path.join(path_img, filename)

    # print(filepath)
    # if filepath in files:
    #     count += 1

    image = cv.imread(filepath, 0)

    return image


def get_face(image, location):
    '''
    Get the face from location object
    :param image: image containing the face
    :param location: location or coordinate of the face
    :return: face image
    '''
    xmin, ymin, width, height = int(location[0]), int(location[1]), int(location[2]), int(location[3])

    xmax = xmin + width
    ymax = ymin + height

    face = image[ymin:ymax, xmin:xmax]

    return face


def set_faces_pipa(path):
    '''
    This method will read images directory, format and sort the data and store in a dump. This will store the faces.
    :param path: path to the parent folder
    :return: None
    '''
    # Insert the sequence of folders <str> in this format: <str1>, <str2>, ...
    path_index = os.path.join(path, 'data', 'PIPA', 'annotations', 'index.txt')

    # if the DUMP folder doesn't exist, create one
    dump_p = os.path.join(path, 'DUMP','PIPA' ,'DUMP_FACE.pkl')
    if not os.path.exists(os.path.dirname(dump_p)):
        os.mkdir(os.path.dirname(dump_p))

    if os.path.exists(dump_p):
        return

    # Get the id dictionary
    id_dict = read_index(path_index)

    # path to train folder in PIPA
    path_img = os.path.join(path, 'data', 'PIPA', 'train')
    files = os.listdir(path_img)

    # This dict will contain face images for each identity
    face_dict = {}

    # iterate over identity and read the face and update the face dict
    for id in id_dict.keys():
        flag = False
        print(id)
        photos_details = id_dict[id]
        for photo in photos_details:

            photo_set, photo_id, subset = photo[0]

            if subset == '1':
                img = get_image(files, path_img, photo_set, photo_id)

                face = get_face(img, photo[1])

                if id not in face_dict.keys():
                    face_dict[id] = []
                else:
                    value = face_dict[id]
                    value.append(face)
                    face_dict[id] = value
            else:
                img = None


    with open(dump_p, 'wb+') as f:
        pickle.dump([id_dict, face_dict], f)



def get_pipa_faces(path):
    '''
    This method will read the DUMP of PIPA faces and return its content
    :param path: path of the parent folder
    :return: face dictionary
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'DUMP')

    with open(os.path.join(path, 'DUMP_faces_PIPA.pkl'), 'rb') as f:
        data = pickle.load(f)

    return data[1], data[0]


if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    face_dict = set_faces_pipa(path)
