import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np
import cyvlfeat
from math import sqrt


def get_faces_hist(dataset, num_identity=5, num_img_per_id=50, resize=100, min_sample = False):
    '''
    This method will return histogram of each image for each identity
    :param dataset: Training or test dataset
    :param num_identity: total number of identity to be used for training/testing
    :param num_img_per_id: max number of images per ID
    :param resize: Size of the images to be used for training
    :return: list of histogram for every image and corresponding labels
    '''
    labels = []
    histograms = []

    if min_sample and num_identity != 0:
        min_count = min([len(dataset[id]) for i, id in enumerate(dataset) if i <= num_identity])
        print("Minimum number of image used: ", min_count)

    for index,identity in enumerate(dataset):
        count_imgs = 0
        # iterate over each identity
        for i, image_p in enumerate(dataset[identity]):
            image = cv.imread(image_p, 0)
            if image is None:
                continue
            # perform histogram equalization before using the image
            image = cv.resize(image,(resize, resize))
            # image = cv.equalizeHist(cv.resize(image,(resize, resize)))
            histograms.append(get_orb_histograms(image,resize))
            labels.append(int(identity))

            count_imgs += 1
            if num_img_per_id != 0 and (count_imgs >= num_img_per_id or (min_sample and count_imgs >= min_count)):
                break

        print(identity)
        if index >= num_identity and num_identity != 0:
            break
    return histograms, labels

def get_surf_desc(image, extended=True):
    '''
    This method will return thr SURF descriptors of the image
    :param image: image
    :param extended: Bool, if true then 128-D descriptors otherwise 64
    :return:
    '''
    surf = cv.xfeatures2d.SURF_create(500)

    # extend descriptor size from 64 to 128
    surf.setExtended(extended)
    kp, des = surf.detectAndCompute(image, None)
    return kp, des

def get_orb_desc(image, image_size, dense=False):
    '''
    This method will return the orb descriptors
    :param image:
    :param image_size:
    :param dense: Bool
    :return:
    '''
    orb = cv.ORB_create(WTA_K=3)

    if dense:
        kp = dense_keypoints(image)
        kp, des = orb.compute(image, kp)
    else:
        kp, des = orb.detectAndCompute(image,None)

    # cv.imshow('file', image)
    # cv.waitKey(0)
    # print(kp,des)

    return kp, des

def get_sift_desc(image, start = 0, dense= True, rootsift = False):
    sift = cv.xfeatures2d.SIFT_create(nOctaveLayers=5,sigma=sqrt(2))

    if dense:
        kp = dense_keypoints(image, start = start)
        kp, des = sift.compute(image, kp)
        if rootsift:
            des /= (des.sum(axis=1, keepdims=True) + 1e-7)
            des = np.sqrt(des)
    else:
        kp, des = sift.detectAndCompute(image,None)

    return np.asarray(kp), des

def dense_keypoints(image, start=0, stride=1):
    '''
    This will return list of dense keypoints
    :param image: image
    :param stride: stride used
    :return: list of keypoints
    '''
    image_size_Y, image_size_X = image.shape
    x = np.arange(start, image_size_X-start, stride)
    y = np.arange(start, image_size_Y-start, stride)

    kp_list = []
    for i in range(len(x)):
        for j in range(len(y)):
            kp = cv.KeyPoint(x[i], y[j], 1, 0)
            if kp_list is None:
                kp_list = [kp]
            else:
                kp_list.append(kp)

    return kp_list

def get_orb_histograms(image, image_size, bins=10):
    '''
    This will return flattened histogram of orb descriptors for an image
    :param image:
    :param image_size:
    :param bins:
    :return:
    '''

    kp, descs = get_orb_desc(image, image_size)

    histograms = []
    for i,desc in enumerate(descs):
        # print(desc)
        hist, bin_edge = np.histogram(desc, bins, range=(0,300) ,density=False)
        histograms.append(hist)

    return np.asarray(histograms).ravel()

def serialize_keypoints(kps):
    kp_s = []
    for kp in kps:
        kp_s.append([kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id])
    return kp_s

def desirialize_keypoints(kps_data):
    kp = []
    for data in kps_data:
        kp.append(cv.KeyPoint(x=data[0][0], y=data[0][1],_size=data[1],_angle=data[2],_response=data[3], _octave=data[4], _class_id=data[5]))

    return np.asarray(kp)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    path1 = os.path.join(path, 'data','lfw','Aaron_Peirsol', )
    path2 = os.path.join(path, 'data', 'lfw','Aaron_Peirsol',)

    # list_files = sorted(os.listdir(path1))
    # list_files2 = os.listdir(path2)

    image1 = cv.imread(os.path.join(path1, 'Aaron_Peirsol_0001.jpg'), 0)
    image2 = cv.imread(os.path.join(path1, 'Aaron_Peirsol_0002.jpg'), 0)
    # image_test = cv.imread(os.path.join(path2, '1_.jpg'), 0)

    resize_img = 100
    image1 = cv.resize(image1, (resize_img, resize_img))
    image2 = cv.resize(image2, (resize_img, resize_img))

    kps, descs = get_sift_desc(image1, start=12)


    # image1_desc = get_surf_desc(image1)
    # image2_desc = get_surf_desc(image2)

    # matches = match_desc(image1_desc, image2_desc)
    # visualize_match(image1, image2, image1_desc, image2_desc, matches)















    # image_test = cv.resize(image_test, (resize_img, resize_img))
    #
    # # image1_desc = get_surf_desc(image1)
    # # image2_desc = get_surf_desc(image2)
    #
    # image1_desc = get_orb_desc(image1, resize_img)
    # image2_desc = get_orb_desc(image2, resize_img)
    # image_test_desc = get_orb_desc(image_test, resize_img)

    # image1_hist = get_orb_histograms(image1, image_size=resize_img)

    #
    # matches = match_desc(image1_desc, image2_desc)
    # visualize_match(image1, image2, image1_desc, image2_desc, matches)
    #
    # matches_test = match_desc(image1_desc, image_test_desc)
    # print(matches_test)
    # visualize_match(image1, image_test, image1_desc, image_test_desc, matches_test)
    #
    # good, match_mask = good_matches(matches)
    # aligned_desc = alignment(good, image1_desc, image2_desc, image1, image2, visualize=True)
    #
    # good_test, match_mask_test = good_matches(matches_test)
    # alignment(good_test, image1_desc, image_test_desc, image1, image_test, visualize=True)
    #
    # # combined_descriptors = combined_desc(image1, image2)
    #
    # print(len(dense_keypoints(resize_img)))

    get_orb_histograms(image1, resize_img)
