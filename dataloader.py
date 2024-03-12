import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# code adapted from the Sound of AI "DL from design to deployment" tutorials

def load_data(data_path):
    """Loads training dataset from json file.

    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    :return mappings (ndarray): Class names
    :return filenames (ndarray): Filenames

    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["Chroma"])
    y = np.array(data["labels"])
    mappings = np.array(data["mappings"])
    filenames = np.array(data["files"])

    print("Dataset loaded!")
    return X, y, mappings, filenames

# this code needs to be adapted so that it can be used with evaluation.py
# i.e., takes one path and only returns test inputs and targets
def prepare_train_set(data_train_path, validation_size):
    """Creates train and validation sets.

    :param data_path (str): Path to json file containing data
    :param validation_size (float): Percentage of train set used for cross-validation
    :param random_seed (int): The random seed for keras, tf, and/or random

    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    """
    # load data
    X, y, _, _ = load_data(data_train_path)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation

def prepare_test_set(data_test_path):
    """Creates test set.

    :param data_path (str): Path to json file containing data

    :return X_test (ndarray): Inputs for the test set
    :return y_test (ndarray): Targets for the test set
    """

    X_test, y_test, _, files = load_data(data_test_path)

    # add an axis to nd array
    X_test = X_test[..., np.newaxis]

    return X_test, y_test, files

