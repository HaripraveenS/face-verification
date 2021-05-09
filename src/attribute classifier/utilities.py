from collections import OrderedDict
import pickle
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn import svm, metrics, model_selection

# Global variables
map_attributes_features = OrderedDict(); output_file_path, output_low_level_path, df_attributes = None, None, None

def initialize_lfw():
    global output_file_path, output_low_level_path, df_attributes
    output_file_path = "../../data/attribute classifier/lfw/"
    output_low_level_path = "../../data/low level/lfw/"
    map_attributes_features_path = output_file_path + "map_attributes_features.txt"
    attributes_path = "../../../LFW/metadata/modified_attributes.txt"
    if not os.path.exists(map_attributes_features_path):
        file_attributes = open(attributes_path, "r"); file_map_attributes_features = open(map_attributes_features_path, "w")
        for attribute in file_attributes.readline().replace("\n", "").split(",")[2:]: file_map_attributes_features.write(attribute + ":" + input("Enter space separated features for the attribute \"%s\": " % (attribute, )) + "\n")
        file_attributes.close(); file_map_attributes_features.close()
    file_map_attributes_features = open(map_attributes_features_path, "r")
    global map_attributes_features
    for attribute, features in [line.split(":") for line in file_map_attributes_features.readlines()]:
        map_attributes_features[attribute] = features.split()
    file_map_attributes_features.close()
    df_attributes = pd.read_csv(attributes_path)
    file_images_under_error = open(output_low_level_path + "images_under_error.txt", "r")
    while True:
        line = file_images_under_error.readline().replace("\n", "")
        if line == "": break
        line = line.split()[0]
        matched_pattern = re.match(r"^([\w-]+?)_(\d+).jpg$", line[line.rindex("/")+1:])
        name, image_num = matched_pattern.groups()
        name = name.replace("_", " ")
        image_num = int(image_num)
        initial_shape = df_attributes.shape
        df_attributes.drop(df_attributes[(df_attributes['person'] == name) & (df_attributes['imagenum'] == image_num)].index, inplace = True)
        final_shape = df_attributes.shape
        assert (final_shape[0] + 1, final_shape[1]) == initial_shape
    file_images_under_error.close()
    assert df_attributes.shape[0] == np.load(output_low_level_path + "rgb_chin.npy").shape[0]

CELEBA_IMAGES = 20000
def initialize_celeba():
    global output_file_path, output_low_level_path, df_attributes
    output_file_path = "../../data/attribute classifier/celeba/"
    output_low_level_path = "../../data/low level/celeba/"
    map_attributes_features_path = output_file_path + "map_attributes_features.txt"
    attributes_path = "../../../CelebA/metadata/list_attr_celeba.csv"
    if not os.path.exists(map_attributes_features_path):
        file_attributes = open(attributes_path, "r"); file_map_attributes_features = open(map_attributes_features_path, "w")
        for attribute in file_attributes.readline().replace("\n", "").split(",")[1:]:
            file_map_attributes_features.write(attribute + ":" + input("Enter space separated features for the attribute \"%s\": " % (attribute, )) + "\n")
        file_attributes.close(); file_map_attributes_features.close()
    file_map_attributes_features = open(map_attributes_features_path, "r")
    global map_attributes_features
    for attribute, features in [line.split(":") for line in file_map_attributes_features.readlines()]:
        map_attributes_features[attribute] = features.split()
    file_map_attributes_features.close()
    df_attributes = pd.read_csv(attributes_path)[:CELEBA_IMAGES]
    file_images_under_error = open(output_low_level_path + "images_under_error.txt", "r")
    while True:
        line = file_images_under_error.readline().replace("\n", "")
        if line == "": break
        line = line.split()[0]
        name = line[line.rindex("/")+1:]
        initial_shape = df_attributes.shape
        df_attributes.drop(df_attributes[df_attributes['image_id'] == name].index, inplace = True)
        final_shape = df_attributes.shape
        assert (final_shape[0] + 1, final_shape[1]) == initial_shape
    file_images_under_error.close()
    assert df_attributes.shape[0] == np.load(output_low_level_path + "rgb_chin.npy").shape[0]

def learn_SVMs():
    global output_low_level_path, df_attributes, map_attributes_features, output_file_path
    for attribute, features in tqdm(map_attributes_features.items()):
        if len(features) == 0: continue
        final_feature_set = np.empty(shape = (df_attributes.shape[0], 0))
        for feature in features:
            feature_array = np.load(output_low_level_path + feature + ".npy")
            final_feature_set = np.append(final_feature_set, feature_array / feature_array.sum(axis = 1).reshape(-1, 1), axis = 1)
        labels = df_attributes[attribute]

        X_train, X_test, y_train, y_test = model_selection.train_test_split(final_feature_set, labels, test_size = 0.3)
        np.save(output_file_path + attribute + "_X_train", X_train)
        np.save(output_file_path + attribute + "_y_train", y_train)
        np.save(output_file_path + attribute + "_X_test", X_test)
        np.save(output_file_path + attribute + "_y_test", y_test)

        svm_obj = svm.SVC()
        grid_values = {'C':[0.01, 0.1, 1, 5, 10, 15, 20, 25, 30, 50]}
        grid_clf_acc = model_selection.GridSearchCV(svm_obj, param_grid = grid_values, scoring = 'accuracy')
        grid_clf_acc.fit(X_train, y_train)
        with open(output_file_path + attribute + "_svm_model.pkl", "wb") as f: pickle.dump(grid_clf_acc, f)

def test_SVMs():
    global map_attributes_features, output_file_path
    for attribute in map_attributes_features.keys():
        with open(output_file_path + attribute + "_svm_model.pkl", "rb") as f: grid_clf_acc = pickle.load(f)
        predicted_output = grid_clf_acc.predict(np.load(output_file_path + attribute + "_X_test.npy"))
        print("----------For attribute %s----------" % (attribute, ))
        print(metrics.classification_report(np.load(output_file_path + attribute + "_y_test.npy"), predicted_output))