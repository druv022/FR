import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np
import face_descriptors


def match_desc(list_1, list_2, type='ORB'):
    des1 = list_1[1]
    des2 = list_2[1]

    # FLANN parameters
    if type == 'ORB':
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
    elif type == 'SURF':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return None
    matches = flann.knnMatch(des1, des2, k=2)
    # print("Matches", matches, len(matches))
    return matches


def good_matches(matches):
    good = []
    matchesMask = []
    if matches is not None:
        matchesMask = [[0, 0] for i in np.arange(len(matches))]
        for i, match in enumerate(matches):
            if len(match) > 1:
                m, n = match
                if m.distance < 0.85 * n.distance:
                    good.append(m)
                    matchesMask[i] = [1, 0]

    return good, matchesMask


def visualize_match(image1, image2, list_1, list_2, matches, matchesMask=None):
    # print(matchesMask)
    if matchesMask is None and matches is not None:
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in np.arange(len(matches))]

        # ratio test as per Lowe's paper
        for i, match in enumerate(matches):
            if isinstance(match, list) and len(match) > 1:
                m, n = match
                if m.distance < 0.85 * n.distance:
                    matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    image3 = cv.drawMatchesKnn(image1, list_1[0], image2, list_2[0], matches, None, **draw_params)
    plt.imshow(image3)
    plt.show()


def alignment(good, list_1, list_2, image1, image2, min_match_count=15, visualize=False):
    kp1, desc1 = list_1
    kp2, desc2 = list_2

    aligned_match_count = 0
    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0, 0.99)
        matchesMask = mask.ravel().tolist()
        print("aligned match")
    else:
        print("Not enough matches")
        return

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    if visualize:
        h, w = image1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        image2 = cv.polylines(image2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        img3 = cv.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()

    if not correct_match(src_pts.reshape(-1, 2) * np.asarray(matchesMask).reshape(-1, 1), dst_pts.reshape(-1, 2)):
        print("THIS IS NOT THE RIGHT IMAGE! (after alignment)")
        return

    return src_pts * np.asarray(matchesMask).reshape(-1, 1, 1)


def correct_match(source_pts, destin_pts):
    # print(source_pts,"\n",dest_pts,)
    # print(source_pts)
    non_zero_idx = np.nonzero(source_pts)
    # print(non_zero_idx)

    src_pts = [source_pts[non_zero_idx[0][i]] for i in range(0, len(non_zero_idx[0]))]
    dst_pts = [destin_pts[non_zero_idx[0][i]] for i in range(0, len(non_zero_idx[0]))]

    # print("SRC:",src_pts,"\nDST:",dst_pts)

    distance = sorted([np.linalg.norm((src_pts[i] - dst_pts[i])) for i in range(len(src_pts))])

    # print("DISTANCE: ",distance)
    mis_match = 0
    dis_ratio = 1
    for i in range(2, len(distance), 2):
        if distance[i - 2] != 0:
            dis_ratio = distance[i] / distance[i - 2]

        # print(dis_ratio, mis_match, len(src_pts))

        if dis_ratio > 1.2 or dis_ratio < 0.8:
            mis_match += 1

        if mis_match >= len(src_pts) / 4:
            return False

    return True


def combined_desc(image1, image2, image_size):
    image1_desc = face_descriptors.get_orb_desc(image1, image_size)
    image2_desc = face_descriptors.get_orb_desc(image2, image_size)

    matches = match_desc(image1_desc, image2_desc)
    good, match_mask = good_matches(matches)
    aligned_kp = alignment(good, image1_desc, image2_desc, image1, image2, visualize=False)

    if aligned_kp is not None:
        aligned_kp = aligned_kp.squeeze(1)
        index = [i for i, kp in enumerate(image1_desc[0]) if kp.pt in aligned_kp]

        return np.asarray(image1_desc[1][index])

    return None


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    path1 = os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '1')
    path2 = os.path.join(path, 'faces', 'ADIENCE_ALIGNED', '13')

    list_files = sorted(os.listdir(path1))
    list_files2 = os.listdir(path2)

    image1 = cv.imread(os.path.join(path1, '1_.jpg'), 0)
    image2 = cv.imread(os.path.join(path1, '2_.jpg'), 0)
    image_test = cv.imread(os.path.join(path2, '1_.jpg'), 0)

    resize_img = 300
    image1 = cv.resize(image1, (resize_img, resize_img))
    image2 = cv.resize(image2, (resize_img, resize_img))
    image_test = cv.resize(image_test, (resize_img, resize_img))

    #
    # # image1_desc = get_surf_desc(image1)
    # # image2_desc = get_surf_desc(image2)
    #
    # image1_desc = get_orb_desc(image1, resize_img)
    # image2_desc = get_orb_desc(image2, resize_img)
    # image_test_desc = get_orb_desc(image_test, resize_img)
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
