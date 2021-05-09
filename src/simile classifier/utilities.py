import numpy as np
from tqdm import tqdm
from sklearn import svm, metrics, model_selection
import pickle

output_file_path, output_low_level_path, list_ref_persons, ref_file_path  = None, None, None, None
dict_facepart_features = {"eyes":["rgb_left_eye", "hsv_left_eye", "grad_mag_left_eye", "grad_orien_left_eye", "rgb_right_eye", "hsv_right_eye", "grad_mag_right_eye", "grad_orien_right_eye"], "nose":["rgb_nose", "hsv_nose", "grad_mag_nose", "grad_orien_nose"], "mouth":["rgb_mouth", "hsv_mouth", "grad_mag_mouth", "grad_orien_mouth"]}

def initialize_celeb_face_recog():
    global output_file_path, output_low_level_path, list_ref_persons, ref_file_path
    output_file_path = "../../data/simile classifier/celeb_face_recog/"
    output_low_level_path = "../../data/low level/celeb_face_recog/"
    ref_file_path = "../../../CelebrityFaceRecognition/reference_faces/"
    file_ref_persons = open(ref_file_path + "reference_faces.txt", "r")
    list_ref_persons = [ref_person.replace("\n", "") for ref_person in file_ref_persons.readlines()]
    file_ref_persons.close()

def learn_SVMs():
    global output_file_path, output_low_level_path, list_ref_persons, ref_file_path
    for ref_person in list_ref_persons:
        for attribute, features in tqdm(dict_facepart_features.items()):
            final_feature_set = np.empty(shape = (np.load(output_low_level_path + features[0] + ".npy").shape[0], 0))
            for feature in features:
                feature_array = np.load(output_low_level_path + feature + ".npy")
                final_feature_set = np.append(final_feature_set, feature_array / feature_array.sum(axis = 1).reshape(-1, 1), axis = 1)
            labels = np.load(ref_file_path + ref_person + "_labels.npy")

            indices = (labels == -1)
            X_train_1, X_test_1, y_train_1, y_test_1 = model_selection.train_test_split(final_feature_set[indices], labels[indices], test_size = 0.3)
            indices = (labels == 1)
            X_train_2, X_test_2, y_train_2, y_test_2 = model_selection.train_test_split(final_feature_set[indices], labels[indices], test_size = 0.15)
            X_train = np.append(X_train_1, X_train_2, axis = 0)
            X_test = np.append(X_test_1, X_test_2, axis = 0)
            y_train = np.append(y_train_1, y_train_2)
            y_test = np.append(y_test_1, y_test_2)
            np.save(output_file_path + ref_person + "_" + attribute + "_X_train", X_train)
            np.save(output_file_path + ref_person + "_" + attribute + "_y_train", y_train)
            np.save(output_file_path + ref_person + "_" + attribute + "_X_test", X_test)
            np.save(output_file_path + ref_person + "_" + attribute + "_y_test", y_test)

            svm_obj = svm.SVC()
            grid_values = {'C':[0.01, 0.1, 1, 5, 10, 15, 20, 25, 30, 50]}
            grid_clf_acc = model_selection.GridSearchCV(svm_obj, param_grid = grid_values, scoring = 'accuracy')
            grid_clf_acc.fit(X_train, y_train)
            with open(output_file_path + ref_person + "_" + attribute + "_svm_model.pkl", "wb") as f: pickle.dump(grid_clf_acc, f)

def test_SVMs():
    global output_file_path
    for ref_person in list_ref_persons:
        for attribute in dict_facepart_features:
            with open(output_file_path + ref_person + "_" + attribute + "_svm_model.pkl", "rb") as f: grid_clf_acc = pickle.load(f)
            predicted_output = grid_clf_acc.predict(np.load(output_file_path + ref_person + "_" + attribute + "_X_test.npy"))
            print("----------For reference person %s and attribute %s----------" % (ref_person, attribute))
            print(metrics.classification_report(np.load(output_file_path + ref_person + "_" + attribute + "_y_test.npy"), predicted_output))