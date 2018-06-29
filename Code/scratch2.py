import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np
import face_descriptors
import pickle
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from sklearn import ensemble, feature_selection
from sklearn.pipeline import  make_pipeline
from sklearn.metrics import classification_report


# USED FOR ALIGNMENT TEST
def get_train_face_desc(training_set, num_train_identity=5, num_train_img_per_id=80, train_percent=0.5, resize=100):
    face_desc_cluster = {}
    face_desc_svm = {}
    count = 0
    for identity in training_set:
        if num_train_img_per_id > len(training_set[identity]):
            continue
        value_cluster = []
        v_svm = []
        for i, image1_p in enumerate(training_set[identity]):
            value_svm = []
            image1 = cv.imread(image1_p, 0)
            if image1 is None:
                continue
            image1 = cv.equalizeHist(cv.resize(image1,(resize, resize)))


            # # comb_desc = face_descriptors.combined_desc(image1, image2)
            # kp, desc = face_descriptors.get_orb_desc(image1)
            # comb_desc = desc
            #
            # if comb_desc is not None:
            #     if len(value_cluster) == 0:
            #         value_cluster = comb_desc
            #     else:
            #         value_cluster = np.concatenate((value_cluster, comb_desc))
            #
            #         # print("@#",type(comb_desc), type(value_cluster))
            #         # print("Cluster:", identity, " i: ",i , " j: ", j )
            # if not isinstance(value_cluster, list):
            #     value = np.zeros((1, value_cluster.shape[1]))
            #     if identity in face_desc_cluster.keys():
            #         value = face_desc_cluster[identity]
            #     value = np.concatenate((value, value_cluster))
            #
            #     face_desc_cluster[identity] = value

            for j, image2_p in enumerate(training_set[identity]):
                image2 = cv.imread(image2_p, 0)
                if image2 is None:
                    # print("SKIP CLuster: ", identity, " ",i, " ", j)
                    continue
                image2 = cv.equalizeHist(cv.resize(image2, (resize, resize)))

                if i < j and i < num_train_img_per_id * train_percent and j < num_train_img_per_id * train_percent:

                    # print(i,j,image1_p, image2_p)
                    comb_desc = face_descriptors.combined_desc(image1, image2, resize)

                    if comb_desc is not None:
                        if len(value_cluster) == 0:
                            value_cluster = comb_desc
                        else:
                            comb_desc_new = np.asarray([i for i in comb_desc if i not in value_cluster])
                            if len(comb_desc_new.shape) > 1:
                                value_cluster = np.concatenate((value_cluster, comb_desc_new))

                            # print("@#",type(comb_desc), type(value_cluster))
                            # print("Cluster:", identity, " i: ",i , " j: ", j )

                if i > num_train_img_per_id * train_percent and j >= num_train_img_per_id * train_percent \
                        and i < num_train_img_per_id and j < num_train_img_per_id:
                    # print(i,j,image1_p, image2_p)

                    comb_desc = face_descriptors.combined_desc(image1, image2, resize)

                    if comb_desc is not None:
                        if len(value_svm) == 0:
                            value_svm = comb_desc
                        else:
                            comb_desc_new = np.asarray([i for i in comb_desc if i not in value_svm])
                            if len(comb_desc_new.shape) > 1:
                                value_svm = np.concatenate((value_svm, comb_desc))
                            # print("SVM:", identity, " i: ", i)

            if v_svm is None:
                v_svm = value_svm
            else:
                v_svm.append(value_svm)

            if not isinstance(value_cluster, list):
                value = np.zeros((1, value_cluster.shape[1]))
                if identity in face_desc_cluster.keys():
                    value = face_desc_cluster[identity]
                value = np.concatenate((value, value_cluster))

                face_desc_cluster[identity] = value
                # print("CLUSTER: ",identity, value.shape)

            # if i > num_train_img_per_id * train_percent and i < num_train_img_per_id:
            #     image1 = cv.imread(image1_p, 0)
            #     if image1 is None:
            #         # print("SKIP SVM: ", identity, " ", i)
            #         continue
            #     # print(i,j,image1_p, image2_p)
            #     kp, desc = face_descriptors.get_orb_desc(image1, resize)
            #     if desc is not None:
            #         if len(value_svm) == 0:
            #             value_svm = desc
            #         else:
            #             value_svm = np.concatenate((value_svm, desc))
            #             # print("SVM:", identity, " i: ", i)
            #
            #     if v_svm is None:
            #         v_svm = value_svm
            #     else:
            #         v_svm.append(value_svm)

        face_desc_svm[identity] = v_svm
        count += 1
        if count >= num_train_identity:
            break

    return face_desc_cluster, face_desc_svm


def visual_vocab(face_desc):
    visual_vocab_cluster = []

    for identity in face_desc:
        visual_vocab_cluster = visual_vocab_cluster + [desc for i, desc in enumerate(face_desc[identity])]


    print("Length of visual vocab before clustering: ",len(visual_vocab_cluster))

    return np.asarray(visual_vocab_cluster)


def clustering_vocab(visual_vocabulary, num_of_vwords=300, verbose=5):
    if len(visual_vocabulary) > num_of_vwords:
        kmeans = KMeans(num_of_vwords, init='k-means++', n_init=3, max_iter=500, verbose=verbose, n_jobs=1,
                        tol=0.000001).fit(visual_vocabulary)
        return kmeans, kmeans.cluster_centers_


def nearest_neighbor(query=None, data=None, tree=None):
    index = None
    if tree is None and data is None:
        print("Please pass data to train KDtree")
    elif tree is None:
        tree = cKDTree(data, copy_data=False)
    else:
        distance, index = tree.query(query, k=1, distance_upper_bound=150)

        # print(distance,"\n",index)

    return tree, index


def get_histogram(kdtree, train_faces_desc, bins=100):
    labels = []
    histogram = []

    for identity in train_faces_desc:
        faces_desc = train_faces_desc[identity]
        # print(len(faces_desc))
        for descs in faces_desc:
            if len(descs) > 2:
                kdtree, index = nearest_neighbor(tree=kdtree, query=descs)
                labels.append(int(identity))
                hist, bin_edge = np.histogram(index, bins, density=True)
                histogram.append(hist)

                # print(histogram, labels)

    return histogram, labels


def get_histogram_test(kdtree, image_desc, bins):
    kdtree, index = nearest_neighbor(tree=kdtree, query=image_desc)
    hist, bin_edge = np.histogram(index, bins, density=True)

    return hist

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    # ADIENCE_ALIGNED, LFW
    with open(os.path.join(path, 'faces', 'ADIENCE_ALIGNED', 'train_test.pkl'), 'rb') as f:
        data = pickle.load(f)
    train_set, test_set = data

    resize_img = 300
    num_id = 100
    num_tr =5

    train_hists, train_labels = face_descriptors.get_faces_hist(train_set,resize=resize_img,num_identity=num_id, num_img_per_id=num_tr)

    print(train_hists, "\n", train_labels)
    print(len(train_hists[0]))

    anova_filter = feature_selection.SelectPercentile(feature_selection.f_classif)
    # c = [2,1.5,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    c = [1]
    for c_value in c:
        clf = SVC(kernel='linear',decision_function_shape='ovo', C=c_value, class_weight='balanced',probability=True,verbose=0)
        clf = make_pipeline(anova_filter,clf)

        # clf = ensemble.AdaBoostClassifier(clf,n_estimators=50)
        # clf = ensemble.BaggingClassifier(clf)
        clf.fit(train_hists, train_labels)

        test_hists, test_labels = face_descriptors.get_faces_hist(test_set,resize=resize_img,num_identity=num_id, num_img_per_id=5)

        # print(len(test_hists[0]))
        pred_labels = clf.predict(test_hists)
        pred_prob = clf.predict_proba(test_hists)
        print(classification_report(test_labels,pred_labels))
        # print("TEST:", clf.predict(histograms))

        # [print(test_labels[i],pred_labels[i],pred_prob[i]) for i in range(len(test_labels))]
        print(c_value,np.count_nonzero([1 if pred_labels[i] == test_labels[i] else 0 for i in range(len(test_labels))]), len(test_labels))





    ##---------------------------- USED FOR ALIGNMENT TEST--------------------------------------------------------------
    # train_percent = 0.3
    # train_faces_cluster, train_faces_svm = get_train_face_desc(train_set, num_train_identity=3, num_train_img_per_id=20,
    #                                                            train_percent=train_percent, resize=resize_img)
    #
    # print(train_faces_cluster.keys(), "\n", train_faces_svm.keys())
    #
    # visual_vocab_cluster = visual_vocab(train_faces_cluster)
    # # print(visual_vocab_cluster.shape)
    #
    # num_of_vwords = 300
    #
    # kmeans, cluster_centers = clustering_vocab(visual_vocab_cluster, num_of_vwords, verbose=5)
    # #
    # # # print(cluster_centers.shape)
    # #
    # kdtree, index = nearest_neighbor(data=cluster_centers)
    #
    # bins = 100
    # histograms, labels = get_histogram(kdtree, train_faces_svm, bins=bins)


    # print( len(labels),labels)

    # clf = SVC(kernel='linear',decision_function_shape='ovr', C=0.9, class_weight='balanced',verbose=5)
    # clf.fit(histograms, labels)
    # print(train_faces_svm.keys(), labels[0],"\n\n")

    # hist_test = get_histogram_test(kdtree, train_faces_svm[labels[0]][0], bins)

    # print(histograms[0], "\n", hist_test)

    # print(train_faces_svm.keys())
    # print("@CHECK: ", labels)
    # # print("TEST:", clf.predict(hist_test.reshape(1,-1)))
    # test_labels = clf.predict(histograms)
    # # print("TEST:", clf.predict(histograms))
    #
    # [print(labels[i], test_labels[i]) for i in range(len(labels))]
    # print(np.count_nonzero([1 if labels[i] == test_labels[i] else 0 for i in range(len(labels))]), len(labels))

    ##------------------------------------------------------------------------------------------------------------------



