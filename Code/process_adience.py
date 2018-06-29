import cv2 as cv
import os
import pickle


def set_faces_adience_aligned(path):
    '''
    This method will read images directory, format and sort the data and store in a dump.
    :param path: Path of parent directory
    :return: None
    '''
    path_m = os.path.join(path, 'data', 'adience', 'aligned')
    list_dir = os.listdir(path_m)

    # if the DUMP folder doesn't exist, create one
    dump_p = os.path.join(path, 'DUMP', 'ADIENCE_ALIGNED','DUMP_faces_adience_aligned.pkl')
    if not os.path.exists(os.path.dirname(dump_p)):
        os.mkdir(os.path.dirname(dump_p))

    if os.path.exists(dump_p):
        return

    # this dict will contain for each identity the path to list of images
    face_dict = {}
    for dir in list_dir:
        images = os.listdir(os.path.join(path_m, dir))

        for image in images:
            image_s = image.split('.')

            # read identity from the file name
            identity = image_s[1]
            # if identity is seen for the first time
            if identity not in face_dict.keys():
                face_dict[identity] = []

            # get previously stored location and update with new data
            value = face_dict[identity]
            value.append(os.path.join(path_m, dir, image))
            face_dict[identity] = value

    # Dump the face dict
    with open(dump_p, 'wb+') as f:
        pickle.dump(face_dict, f)


def get_adience_aligned_faces(path):
    '''
    This method will read the DUMP of aligned faces and return its content
    :param path: path of the parent folder
    :return: face dictionary
    '''
    path = os.path.join(path, 'DUMP')

    with open(os.path.join(path, 'DUMP_faces_adience_aligned.pkl'), 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    face_dict = set_faces_adience_aligned(path)
