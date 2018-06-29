import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    path1 = os.path.join(path, 'faces','ADIENCE_ALIGNED','4')
    path2 = os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '2')

    print(path)
    list_files= sorted(os.listdir(path1))
    list_files2 = os.listdir(path2)

    img = cv.imread(os.path.join(path1,'7_.jpg'),0)
    # cv.imshow('face1',img)
    # cv.waitKey(0)
    img2 = cv.imread(os.path.join(path1, '8_.jpg'),0)
    # cv.imshow('face2',img2)
    img_t = cv.imread(os.path.join(path2, '2_.jpg'), 0)
    # cv.imshow('face_t',img_t)
    # cv.waitKey(0)

    surf = cv.xfeatures2d.SURF_create(200)

    # extend descriptor size from 64 to 128
    surf.setExtended(True)
    kp1, des1 = surf.detectAndCompute(img, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    kp3, des3 = surf.detectAndCompute(img_t, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    matches2 = flann.knnMatch(des1, des3, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in np.arange(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.85 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(img, kp1, img2, kp2, matches, None, **draw_params)
    img4 = cv.drawMatchesKnn(img, kp3, img_t, kp3, matches2, None, **draw_params)

    plt.imshow(img3)
    plt.show()
    plt.imshow(img4)
    plt.show()

    # print(len(kp))
    # img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    # plt.imshow(img2)

    plt.show()

    print(surf.descriptorSize())