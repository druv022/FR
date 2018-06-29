import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np
from cyvlfeat.gmm import gmm
from cyvlfeat.fisher import fisher
import face_align
import dlib
import face_descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import whiten
import data
from sys import getsizeof
import pickle
from sklearn.metrics import classification_report, roc_curve, auc
import process_lwf
import random


def embed_spatial_info(kps, descs, image_size):
    h, w = image_size

    descs_new = []
    # print("Image shape: ",image_size,"w: ", w,"h: ",h)
    for i, k_point in enumerate(kps):
        x, y = k_point.pt[0], k_point.pt[1]

        descs_new.append(np.asarray(descs[i].tolist() + [x / w - 0.5] + [y / h - 0.5]))

    return np.asarray(descs_new, dtype="float32")


def initialize_w(fv, p=128):
    fv_list = []
    for item in fv:
        fv_list.append(item[0])
        fv_list.append(item[1])

    print(np.asarray(fv_list).shape)
    fv_pca = PCA(n_components=128, whiten=True).fit_transform(np.asarray(fv_list).transpose())

    w = fv_pca.transpose()
    return w


def distance_img(fv_1, fv_2, w):
    diff = fv_1 - fv_2

    a = np.matmul(diff.reshape(-1, 1).transpose(), w.transpose())
    b = np.matmul(w, diff)
    c = np.matmul(a, b)

    return c


def get_align_face(image, path):
    path_to_pred = os.path.join(path, "code", "shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_pred)

    align_img = face_align.FaceAligner(predictor, desiredFaceHeight=160, desiredFaceWidth=125)
    rects = detector(image, 2)
    for rect in rects:
        image = align_img.align(image, rect)
        break

    return image

def get_gmm(kps, descs, image_size):
    print("GMM started---")

    # descs = StandardScaler().fit_transform(descs)

    pca = PCA(n_components=64)
    descs = pca.fit_transform(descs)

    descs = embed_spatial_info(kps, descs, image_size)

    data_mcp = gmm(descs, n_clusters=512, init_mode='rand',max_num_iterations=100)

    means = data_mcp[0].transpose()
    covariance = data_mcp[1].transpose()
    priors = data_mcp[2]

    return means, covariance, priors

def get_fv(image_size, means, covariance, priors,image= None, sift_dict=None, image_p =  None):
    if sift_dict is None or image_p is None or image_p not in sift_dict.keys():
        kps, descs = face_descriptors.get_sift_desc(image, start=0)
    else:
        kps_data, descs = sift_dict[image_p]
        kps = face_descriptors.desirialize_keypoints(kps_data)

    # descs = StandardScaler().fit_transform(descs)

    pca = PCA(n_components=64)
    descs = pca.fit_transform(descs)

    descs = embed_spatial_info(kps, descs, image_size)

    # print(descs.transpose().shape,priors.shape, means.shape, covariance.shape)

    # D: dimension(SIFT 128-64 dim),n: number of descriptors per image, N: number of GMM, F: Features (SIFT 128-64 dim)
    # fisher([D,n],[F,N],[F,N],[N]) accepts dim
    fv_encoding = fisher(descs.transpose(), means, covariance, priors, normalized=True, square_root=True, fast=True,
                         improved=True)

    return fv_encoding


def get_trainset_sift(train_set, path):
    kps = []
    descs = []
    count = 0
    print("Getting SIFT desc of images. Processing two image per counter")

    sift_dict = {}
    # sift_dict_p = os.path.join(path,'DUMP','LFW','sift_dict.pkl')
    # if os.path.exists(sift_dict_p):
    #     with open(sift_dict_p,'rb') as f:
    #         sift_dict = pickle.load(f)

    for exmpl in train_set:
        count += 1
        print("SIFT Counter: ",count)
        image1_p = exmpl[1][0]
        image2_p = exmpl[1][1]

        if image1_p not in sift_dict.keys():
            image1 = cv.imread(image1_p, 0)
            image1 = get_align_face(image1, path)

            if image1 is None:
                print("Missed")
            kp1, descs1 = face_descriptors.get_sift_desc(image1)
            # sift_dict[image1_p] = [face_descriptors.serialize_keypoints(kp1), descs1]
        else:
            kp1_data, descs1 = sift_dict[image1_p]
            kp1 = face_descriptors.desirialize_keypoints(kp1_data)

        if image2_p not in sift_dict.keys():
            image2 = cv.imread(image2_p, 0)
            image2 = get_align_face(image2, path)

            if image2 is None:
                print("Missed")
            kp2, descs2 = face_descriptors.get_sift_desc(image2)
            # sift_dict[image2_p] = [face_descriptors.serialize_keypoints(kp2), descs2]
        else:
            kp2_data, descs2 = sift_dict[image2_p]
            kp2 = face_descriptors.desirialize_keypoints(kp2_data)

        index1 = np.arange(0, kp1.shape[0])
        index1 = np.random.permutation(index1)
        kp1, descs1 = kp1[index1[0:250]], descs1[index1[0:250]]

        index2 = np.arange(0, kp2.shape[0])
        index2 = np.random.permutation(index2)
        kp2, descs2 = kp2[index2[0:250]], descs2[index2[0:250]]

        kps = kps + kp1.tolist() + kp2.tolist()
        descs = descs + descs1.tolist() + descs2.tolist()

    print("Total number of descs: ", len(descs))

    # with open(sift_dict_p,'wb') as f:
    #     pickle.dump(sift_dict,f)

    index = np.arange(len(kps))

    return np.asarray(kps)[index], np.asarray(descs)[index], sift_dict


def get_fisher_set(data_set, image_size, means, covariance, priors, sift_dict):
    #explm = data_set#[0:40] + train_set[1110:1150]
    fv_encoded = []
    labels = []
    count = 0
    print("Generating Fisher Vector for images. Processing two image per count")
    for exmpl in data_set:
        count += 1
        print("Fisher Counter: ",count)
        labels.append(exmpl[0])
        image1_p = exmpl[1][0]
        image2_p = exmpl[1][1]

        if sift_dict is None or image1_p not in sift_dict.keys():
            image1 = cv.imread(image1_p, 0)
            image1 = get_align_face(image1, path)
        else:
            image1 = None

        if sift_dict is None or image2_p not in sift_dict.keys():
            image2 = cv.imread(image2_p, 0)
            image2 = get_align_face(image2, path)
        else:
            image2 = None

        fv_1 = get_fv(image_size, means, covariance, priors, image=image1, sift_dict=sift_dict,image_p=image1_p)
        fv_2 = get_fv(image_size, means, covariance, priors, image=image2, sift_dict=sift_dict,image_p=image2_p)
        fv_encoded.append([fv_1, fv_2])

    return fv_encoded, labels


def training(train_set, image_size, path, num_iter=1000, gamma_1=0.5, gamma_2=10.0, reuse = False, val_i = -1, test_set = None):
    print("Begin Training--------------------")
    if not reuse:
        kps, descs, sift_dict = get_trainset_sift(train_set, path)

        means, covariance, priors = get_gmm(kps, descs, image_size)

        # [N images [fv_image1, fv_image2]]
        fv_encoded, labels = get_fisher_set(train_set, image_size, means, covariance, priors, sift_dict)

        with open('fv_encoded'+str(val_i)+'.pkl', 'wb') as f:
            pickle.dump([fv_encoded, labels, means, covariance, priors], f)
    else:
        sift_dict_p = os.path.join(path,"DUMP","LFW","sift_dict.pkl")
        if os.path.exists(sift_dict_p):
            with open(sift_dict_p,'rb') as f:
                sift_dict = pickle.load(f)
        else:
            sift_dict =None
        with open('fv_encoded'+str(val_i)+'.pkl', 'rb') as f:
            fv_encoded, labels, means, covariance, priors = pickle.load(f)

    w = initialize_w(fv_encoded, p=128)
    b = 800

    update = 0
    loss_list = []
    acc_list = []
    iter_list = []
    b_list = []
    loss = 0
    for iter_count in range(num_iter):
        if iter_count % 300 == 0:
            print("------------------------------------------------------------------------------")
            print("Training iteration: ", iter_count)
            print("------------------------------------------------------------------------------")


        i = random.randint(0,len(fv_encoded)-1)

        if labels[i] == 1:
            y = 1
        else:
            y = -1

        distance = distance_img(fv_encoded[i][0], fv_encoded[i][1], w)
        diff = fv_encoded[i][0] - fv_encoded[i][1]
        diff = diff.reshape(-1, 1)
        condition_value = y * (b - distance)
        if condition_value <= 1:
            var1 = np.matmul(w, diff)
            var2 = np.matmul(var1, diff.transpose())
            w = w - gamma_1 * y * var2
            b = b + gamma_2 * y
            update += 1

        if 1 - condition_value > 0:
            loss += 1 - condition_value

        if iter_count % 50 == 0:
            print("Loss after iter: ", iter_count, " is ", loss/(update+1e-7))
            loss_list.append(loss/(update+1e-7))
            print("B: ", b)
            iter_list.append(iter_count)
            acc_list.append(checkpoint(test_set,sift_dict,image_size,w,b,means,covariance,priors))
            b_list.append(b)


    with open("Learned_param"+str(val_i)+".pkl", "wb") as f:
        pickle.dump([(w, b),(means,covariance, priors)],f)

    return w, b, means, covariance, priors, sift_dict, [iter_list, acc_list, loss_list, b_list]

def test(test_set, sift_dict,image_size,w = None, b = None, means = None, covariance = None, priors = None, reuse=False, train_set=False, val_i = -1):

    if means is None:
        with open("Learned_param"+str(val_i)+".pkl", "rb") as f:
            data = pickle.load(f)
            w,b = data[0]
            means, covariance,priors = data[1]

    if not reuse:
        fv_encoding, true_labels = get_fisher_set(test_set, image_size,means, covariance, priors,sift_dict)
        with open("fv_encoding_test"+str(val_i)+".pkl", "wb") as f:
            pickle.dump([fv_encoding, true_labels], f)
    else:
        if not train_set:
            print("Run test set")
            with open("fv_encoding_test"+str(val_i)+".pkl", "rb") as f:
                fv_encoding, true_labels = pickle.load(f)
        else:
            print("Run train set")
            with open("fv_encoded"+str(val_i)+".pkl", 'rb') as f:
                fv_encoding, true_labels, means, covariance, priors = pickle.load(f)

    pred_labels = []
    score = []
    for i,fvs in enumerate(fv_encoding):
        distance = distance_img(fvs[0], fvs[1], w)
        # print(distance,b,true_labels[i])

        if distance > b:
            pred_labels.append(0)
            score.append(distance-b)
        else:
            pred_labels.append(1)
            score.append(distance-b)

        # print(true_labels[i], pred_labels[i])

    return true_labels, pred_labels, score

def crossvalidate(dataset, image_size):
    count_data = len(dataset)
    validation_set = 10

    exmpls_per_set = int(count_data/validation_set)

    roc_auc = []
    roc = []

    for i in range(validation_set):
        if i == 0:
            train_set = dataset[exmpls_per_set:exmpls_per_set*validation_set-1]
            test_set = dataset[0:exmpls_per_set-1]
        elif i == validation_set-1:
            train_set = dataset[0:exmpls_per_set*(validation_set-1)-1]
            test_set = dataset[exmpls_per_set*(validation_set-1):exmpls_per_set*validation_set-1]
        else:
            train_set = dataset[0:exmpls_per_set*i-1] + dataset[exmpls_per_set*(i+1):exmpls_per_set*validation_set-1]
            test_set = dataset[exmpls_per_set*i:exmpls_per_set*(i+1) - 1]

        w, b, means, covariance, priors, sift_dict, loss_list = training(train_set, path,image_size, num_iter=5, gamma_1=0.05, gamma_2=2,
                                                   reuse=False,val_i=i)

        true_labels, pred_labels, score = test(test_set, sift_dict,image_size,reuse=False, train_set=False, val_i=i)
        print(classification_report(true_labels, pred_labels))
        fpr, tpr, threshold = roc_curve(true_labels,score)

        roc.append([fpr, tpr, threshold])

        print(auc(fpr, tpr))
        roc_auc.append(auc(fpr, tpr))

        plt.plot(tpr, fpr,label="Set " + str(i))

    with open("result.pkl", 'wb') as f:
        pickle.dump([roc, roc_auc], f)

    plt.show()

def checkpoint(test_set, sift_dict,image_size,w, b, means, covariance, priors):

    true_labels, pred_labels, score = test(test_set,sift_dict,image_size,w = w, b = b, means = means, covariance = covariance, priors = priors, reuse=True)

    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] == pred_labels[i]:
            correct += 1

    return correct/len(true_labels)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    image_size = [160, 125]
    # dataset = process_lwf.get_benchmark(path)
    # crossvalidate(dataset,image_size)

    train_set, test_set = data.get_dataset('LFW', path)

    # used as reference, changing here will not change the image size implicitly
    image_size = [160, 125]
    gamma_1 = 0.09
    gamma_2 = 0

    w, b, means, covariance, priors, sift_dict, data = training(train_set,image_size,path,num_iter=10000,gamma_1=gamma_1,gamma_2=gamma_2,reuse=True, test_set=test_set)

    iter_list, acc_list, loss_list, b_list = data

    plt.plot(iter_list,acc_list,label="Accuracy for gamma 1 " + str(gamma_1) + " and gamma 2 " + str(gamma_2))
    plt.title("Accuracy for gamma 1= " + str(gamma_1) + " and gamma 2= " + str(gamma_2))
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy ")
    plt.show()

    plt.plot(iter_list, loss_list, label="Loss for gamma 1 " + str(gamma_1) + " and gamma 2 " + str(gamma_2))
    plt.title("Loss for gamma 1= " + str(gamma_1) + " and gamma 2= " + str(gamma_2))
    plt.xlabel("Iteration")
    plt.ylabel("Loss ")
    plt.show()

    plt.plot(iter_list, b_list, label="b/threshold for learning rate gamma 2 " + str(gamma_2) + " and " + " gamma 1 " + str(gamma_1))
    plt.title("b/threshold for learning rate gamma 2= " + str(gamma_2) + " and " + " gamma 1= " + str(gamma_1))
    plt.xlabel("Iteration")
    plt.ylabel("b/threshold")
    plt.show()






    # true_labels, pred_labels, score = test(test_set,sift_dict,image_size,reuse=True, train_set=True)
    #
    # print(classification_report(true_labels, pred_labels))
    # fpr, tpr, threshold = roc_curve(true_labels,score)
    #
    # print(auc(fpr, tpr))
    #
    # plt.plot(tpr, fpr)
    #
    # true_labels_, pred_labels_, score_ = test(test_set, sift_dict,image_size,reuse=True)
    # print(classification_report(true_labels_, pred_labels_))
    # fpr_, tpr_, threshold_ = roc_curve(true_labels_, score_)
    #
    # print(auc(fpr_, tpr_))
    #
    # correct = 0
    # for i in range(len(true_labels_)):
    #     if true_labels_[i] == pred_labels_[i]:
    #         correct += 1
    #
    # print (correct/len(true_labels_))
    #
    # plt.plot(tpr_, fpr_)
    # plt.show()

    # kps, descs = get_trainset_sift(train_set[0:5], path)
    #
    # means, covariance, priors = get_gmm(kps, descs,image_size)
    #
    # # [N images [fv_image1, fv_image2]]
    # fv_encoded = get_fisher_set(train_set, means, covariance, priors)
    #
    # with open('fv_encoded.pkl','wb') as f:
    #     pickle.dump(fv_encoded,f)
    # with open('fv_encoded.pkl', 'rb') as f:
    #     fv_encoded = pickle.load(f)
    #
    # w = initialize_w(fv_encoded, p=128)
    #
    # distance = distance_img(fv_encoded[0][0], fv_encoded[0][1], w)
    # diff = fv_encoded[0][0] - fv_encoded[0][1]
    # diff = diff.reshape(-1, 1)
    # b = 1.1
    # gamma_1 = 0.5
    # y = 1
    # a = np.matmul(w, diff)
    # b = np.matmul(a, diff.transpose())
    # w = gamma_1 * y * b

    # print(w.shape)














    # path1 = os.path.join(path, 'data', 'lfw', 'Aaron_Peirsol', )
    #
    # path_to_pred = os.path.join(path, "code", "shape_predictor_68_face_landmarks.dat")
    #
    # image1 = cv.imread(os.path.join(path1, 'Aaron_Peirsol_0001.jpg'), 0)
    # image2 = cv.imread(os.path.join(path1, 'Aaron_Peirsol_0004.jpg'), 0)
    # image = cv.resize(image,(300,300))

    # image1 = get_align_face(image1, path)
    # image2 = get_align_face(image2, path)

    # fv_encoding1 = get_fv(image1)
    # fv_encoding1 = fv_encoding1.reshape(-1,1)

    # print(fv_encoding1.shape)
    # w = initialize_w(fv_encoding1.transpose())

    # fv_encoding2 = get_fv(image2)

    # distance_img(fv_encoding1, fv_encoding2,w)
