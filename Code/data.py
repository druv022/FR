import process_adience
import process_pipa
import process_lwf
import os
import faces
import face_descriptors
from sklearn.svm import SVC
import pickle
import numpy as np
import cv2 as cv
from sklearn import ensemble, feature_selection
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_curve, auc


def get_dataset(dataset_name, path, min_img=5, ratio=0.5, recog_type='pair_test'):
    '''

    :param dataset_name: name of the dataset ,'ADIENCE_ALIGNED', 'LWF', 'PIPA'
    :param path: path to the parent folder
    :param ratio: The ratio between Train and Test set. This is used only when the train_test.pkl is not created.
    :return:
    '''
    # Check if the train_test file exist. this file contains reference/pointer to test and train data
    train_test_p = os.path.join(path, 'DUMP', dataset_name, 'train_test.pkl')

    print(train_test_p, os.path.exists(train_test_p))

    face_folder = os.path.join(path, 'Faces')
    if not os.path.exists(train_test_p):

        if dataset_name == 'ADIENCE_ALIGNED':
            # this will create a dump folder and store the references without segmenting
            process_adience.set_faces_adience_aligned(path)

            face_dict = process_adience.get_adience_aligned_faces(path)

            faces_p = os.path.join(face_folder, dataset_name)

            if not os.path.exists(faces_p):
                faces.dump_face_into_folder(face_dict, dataset_name, face_folder)

            faces.prepare_sets(path, faces_p, dataset_name, min_train_img=min_img, tr_percnt=ratio)

        elif dataset_name == 'LFW':
            process_lwf.set_faces_lfw(path)

            face_dict = process_lwf.get_lfw_faces(path)

            face_folder = os.path.join(path, 'data', 'lfw')
            faces_p = os.path.join(face_folder, dataset_name)
            # # This part is not required if face is not to be detected once again. LFW dataset already uses the face
            # # detector for to collect the faces and reduces the number of images after detecting again.
            # if not os.path.exists(faces_p):
            #     faces.dump_face_into_folder(face_dict, dataset_name, face_folder)

            if recog_type == 'pair_test':
                process_lwf.split_lfw(path)
            else:
                faces.prepare_sets(path, faces_p, dataset_name, min_train_img=min_img, tr_percnt=ratio)

        elif dataset_name == 'PIPA':
            process_pipa.set_faces_pipa(path)

            face_dict = process_pipa.get_pipa_faces(path)

            faces_p = os.path.join(face_folder, dataset_name)

            if not os.path.exists(faces_p):
                faces.dump_face_into_folder(face_dict, dataset_name, face_folder)

            faces.prepare_sets(path, faces_p, dataset_name, min_train_img=min_img, tr_percnt=ratio)

            # faces_p = os.path.join(face_folder, dataset_name)
            #
            # if not os.path.exists(faces_p):
            #     faces.dump_face_into_folder(face_dict, dataset_name, face_folder)

            # faces.prepare_sets(faces_p, min_train_img=0, tr_percnt=ratio)

    with open(train_test_p, 'rb') as f:
        data = pickle.load(f)

    return data


def train_ver(train_set, resize_img=300, num_id=5, num_img_id=10, min_sample=False):
    '''
    This method will train SVM classifier on given training set
    :param resize_img: image resize to 300 by default
    :param no_id: number of identity to be tested
    :param num_img_id: number of images per identity to be trained on
    :return: the trained SVM model
    '''

    print("TRAINING BEGIN for ", num_id, " ID's", num_img_id, " images each")

    train_hists, train_labels = face_descriptors.get_faces_hist(train_set, resize=resize_img, num_identity=num_id,
                                                                num_img_per_id=num_img_id, min_sample=min_sample)

    anova_filter = feature_selection.SelectPercentile(feature_selection.f_classif)

    # Initialize SVM
    clf = SVC(kernel='linear', decision_function_shape='ovo', C=1, class_weight='balanced', probability=True, verbose=5)
    clf = make_pipeline(anova_filter, clf)
    # fit the model
    clf.fit(train_hists, train_labels)

    return clf


def train_pair(train_set, resize_img=300, num_id=5, num_img_id=10, min_sample=False, params=None):
    train_labels = []
    train_hists = []

    for item in train_set:
        print(item)
        train_labels.append(item[0])
        image1_p = item[1][0]
        image2_p = item[1][1]

        image1 = cv.imread(image1_p, 0)
        image2 = cv.imread(image2_p, 0)

        image1_hist = face_descriptors.get_orb_histograms(image1, resize_img)
        image2_hist = face_descriptors.get_orb_histograms(image2, resize_img)

        train_hists.append(np.concatenate((image1_hist, image2_hist)))

    anova_filter = feature_selection.SelectPercentile(feature_selection.f_classif)

    if params is not None:
        c, gamma = params
        models = []
        for c_value in c:
            for g_value in gamma:
                # Initialize SVM
                print("Training SVM: c ", c_value, ",gamma ", g_value)
                clf = SVC(kernel='rbf', decision_function_shape='ovr', C=c_value, gamma=g_value,
                          class_weight='balanced', probability=True, verbose=5)
                clf = make_pipeline(anova_filter, clf)
                # fit the model
                clf.fit(train_hists, train_labels)
                models.append([clf, c_value, g_value])
        return models
    else:
        clf = SVC(kernel='linear', decision_function_shape='ovr', C=1, class_weight='balanced',
                  probability=True, verbose=5)
        clf = make_pipeline(anova_filter, clf)
        # fit the model
        clf.fit(train_hists, train_labels)

        return clf


def test_model_set(test_set, model, resize_img=300, num_id=5, num_img_per_id=5):
    test_hists, test_labels = face_descriptors.get_faces_hist(test_set, resize=resize_img, num_identity=num_id,
                                                              num_img_per_id=num_img_per_id)

    pred_labels = model.predict(test_hists)

    return test_labels, pred_labels


def test_model_pair(test_set, model, resize_img=300, num_id=5, num_img_per_id=5):
    test_labels = []
    test_hists = []
    for i, item in enumerate(test_set):
        # print(i, item)
        test_labels.append(item[0])

        image1_p = item[1][0]
        image2_p = item[1][1]

        image1 = cv.imread(image1_p, 0)
        image2 = cv.imread(image2_p, 0)

        image1_hist = face_descriptors.get_orb_histograms(image1, resize_img)
        image2_hist = face_descriptors.get_orb_histograms(image2, resize_img)

        test_hists.append(np.concatenate((image1_hist, image2_hist)))

    pred_labels = model.predict(test_hists)
    pred_prob = model.predict_proba(test_hists)

    return test_labels, pred_labels, pred_prob


def save_model(model, name, path):
    model_path = os.path.join(path, 'Models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    with open(os.path.join(model_path, name + '.pkl'), 'wb+') as f:
        pickle.dump(model, f)

    print("Model asved at ", str(model_path))


def load_model(name, path):
    model_path = os.path.join(path, 'Models', name + '.pkl')
    if not os.path.exists(model_path):
        print("Model doesn't exist")
        return

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    return data


def get_split_data(dataset):
    for identity in dataset:
        label = identity
        for image_p in dataset[identity]:
            image = cv.imread(image_p, 0)
            yield [image, label]


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    dataset = 'LFW'
    train_set, test_set = get_dataset(dataset, path, min_img=0)

    # 'verification': identify one id from the lot
    # 'pair_test': pair of image are of the same person or different/True(1) or False(0) : Only for 'LFW'
    recog_type = 'pair_test'
    num_id = 0
    c = [2, 1.5, 1, 0.9, 0.8, 0.5]
    gamma = [10, 5, 2, 1, 0.9, 0.8, 0.5, 0.1, 0.01]

    if recog_type == 'verification':
        # will require further testing after modification!
        clf_model = train_ver(train_set, resize_img=250, num_id=num_id, num_img_id=7, min_sample=False)
        test_labels, pred_labels = test_model_set(test_set, clf_model, num_id=num_id, num_img_per_id=5)
    elif recog_type == 'pair_test':
        params = [c, gamma]
        clf_model = train_pair(train_set, resize_img=250, num_id=num_id, num_img_id=0, min_sample=False,params=params)
        # clf_model = load_model('All_models', path)

        if params is not None:
            save_model(clf_model, 'All_ND_models', path)
            roc_models = []
            roc_auc_models = []
            for clf in clf_model:
                print("Test Models: c = ", clf[1], " ,gamma ", clf[2])
                test_labels, pred_labels, pred_prob = test_model_pair(test_set, clf[0], num_id=num_id, num_img_per_id=5)
                # [print(test_labels[i], pred_labels[i], pred_prob[i]) for i in range(len(test_labels))]
                print(classification_report(test_labels, pred_labels))

                # print(test_labels, "\n pred prob",pred_prob[:,1])
                fpr, tpr, _ = roc_curve(test_labels, pred_prob[:,1])
                roc_auc = auc(fpr, tpr)

                # print("FRP: ",fpr, tpr)
                roc_models.append([fpr, tpr])
                roc_auc_models.append(roc_auc)

            with open('roc_dump.pkl','wb') as f:
                pickle.dump([roc_models, roc_auc_models], f)
                # save_model(clf_model,'Test1',path)



                # print(classification_report(test_labels, pred_labels))

                # [print(test_labels[i], pred_labels[i]) for i in range(len(test_labels))]
                # print(np.count_nonzero([1 if pred_labels[i] == test_labels[i] else 0 for i in range(len(test_labels))]),
                #       len(test_labels))
