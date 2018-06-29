import cv2 as cv
import os
import pickle
from faces import detect_face
import json


def set_faces_lfw(path):
    '''
    This method will read images directory, format and sort the data and store in a dump
    :param path: path to parent directory
    :return: None
    '''
    path_m = os.path.join(path, 'data', 'lfw')
    list_names = os.listdir(path_m)
    names_dict = {}

    # if the DUMP folder doesn't exist, create one
    dump_p = os.path.join(path, 'DUMP','LFW','DUMP_faces_LFW.pkl')
    if not os.path.exists(os.path.dirname(os.path.dirname(dump_p))):
        os.mkdir(os.path.dirname(os.path.dirname(dump_p)))
    if not os.path.exists(os.path.dirname((dump_p))):
        os.makedirs(os.path.dirname(dump_p))

    if os.path.exists(dump_p):
        return

    # this dict will contain for each identity the path to list of images
    face_dict = {}
    # this variable is used to replace peoples names to numeric value
    count_identity = 0
    for identity in list_names:
        count_identity += 1
        names_dict[identity] = count_identity
        if count_identity not in face_dict.keys():
            face_dict[count_identity] = [str(identity)]

        images_path = os.path.join(path_m, identity)
        images_list = os.listdir(images_path)

        value = face_dict[count_identity]

        # for each image if a face is detectable, only then store the path. This is to ensure real time
        for image in images_list:
            image_p = os.path.join(images_path, image)
            # image = cv.imread(image_p, 0)
            # face, location = detect_face(image)
            # if face is not None:
            value.append(image_p)

        face_dict[count_identity] = value
        # print("SET LFW",count_identity)

    # Dump face dict
    with open(dump_p, 'wb+') as f:
        pickle.dump(face_dict, f)

    with open(os.path.join(path, 'DUMP','LFW' ,'LFW_names.pkl'), 'wb+') as f:
        pickle.dump(names_dict, f)


def get_lfw_faces(path):
    '''
    This method will read the DUMP of lfw faces and return its content
    :param path: path of the parent folder
    :return: face dictionary
    '''

    path = os.path.join(path, 'DUMP','LFW')

    with open(os.path.join(path, 'DUMP_faces_LFW.pkl'), 'rb') as f:
        data = pickle.load(f)

    return data


def split_lfw(path):
    path_to_dump = os.path.join(path, 'DUMP','LFW')
    with open(os.path.join(path_to_dump, 'LFW_names.pkl'), 'rb') as f:
        names_dict = pickle.load(f)

    path_to_split_txt = os.path.join(path, 'data', 'lfw_train_test')

    path_to_data = os.path.join(path, 'data', 'lfw')

    train_set = []
    test_set = []

    with open(os.path.join(path_to_split_txt, 'pairsDevTrain.txt'), 'r') as f:
        # print(f.readlines())
        for line in f.readlines():
            data = line.split()
            ret_data = filter_pair(data,path_to_data,names_dict)
            if ret_data is not None:
                train_set.append(ret_data)

    with open(os.path.join(path_to_split_txt, 'pairsDevTest.txt'), 'r') as f:
        for line in f.readlines():
            data = line.split()
            ret_data = filter_pair(data,path_to_data,names_dict)
            if ret_data is not None:
                test_set.append(ret_data)

    with open(os.path.join(path_to_dump, 'train_test.pkl'), 'wb+') as f:
        pickle.dump([train_set, test_set], f)

def filter_pair(data, path_to_data, names_dict):
    data_length = len(data)
    if data_length == 0:
        return
    elif data_length == 3:
        identity = names_dict[data[0]]

        path_to_faces = os.path.join(path_to_data, data[0])
        list_files = os.listdir(path_to_faces)

        num_img = len(list_files)
        if num_img < max(int(data[1]), int(data[2])):
            return
        return [1, [os.path.join(path_to_faces, data[0]+'_'+str(data[1]).zfill(4))+'.jpg', os.path.join(path_to_faces, data[0]+'_'+str(data[2]).zfill(4))+'.jpg']]

    elif data_length == 4:
        identity1 = names_dict[data[0]]
        identity2 = names_dict[data[2]]

        path_to_faces1 = os.path.join(path_to_data, data[0])
        path_to_faces2 = os.path.join(path_to_data, data[2])

        list_files1 = os.listdir(path_to_faces1)
        if len(list_files1) < int(data[1]):
            return

        list_files2 = os.listdir(path_to_faces2)
        if len(list_files2) < int(data[3]):
            return

        return [0, [os.path.join(path_to_faces1,data[0]+'_'+str(data[1]).zfill(4))+'.jpg', os.path.join(path_to_faces2, data[2]+'_'+str(data[3]).zfill(4))+'.jpg']]


def get_benchmark(path):
    path_to_dump = os.path.join(path, 'DUMP', 'LFW')
    with open(os.path.join(path_to_dump, 'LFW_names.pkl'), 'rb') as f:
        names_dict = pickle.load(f)

    path_to_split_txt = os.path.join(path, 'data', 'lfw_train_test')

    path_to_data = os.path.join(path, 'data', 'lfw')

    dataset = []
    with open(os.path.join(path_to_split_txt, 'pairs_benchmark.txt'), 'r') as f:
        # print(f.readlines())
        for line in f.readlines():
            data = line.split()
            ret_data = filter_pair(data, path_to_data, names_dict)
            if ret_data is not None:
                dataset.append(ret_data)

    return dataset

if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    set_faces_lfw(path)
