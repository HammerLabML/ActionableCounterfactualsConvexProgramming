# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.utils import shuffle


def covariance_to_correlation(cov):
    t = np.linalg.inv(np.sqrt(np.diag(np.diagonal(cov))))
    return t @ cov @ t


def get_delta_overlap(a, b):
    r = 0
    for i in range(a.shape[0]):
        if a[i] != 0 and b[i] != 0:
            r += 1
        elif a[i] == 0 and b[i] == 0:
            r += 1

    return r


def load_data_iris():
    X, y = load_iris(return_X_y=True)

    return X, y


def load_data_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)

    return X, y


def load_data_wine():
    X, y = load_wine(return_X_y=True)

    return X, y


def load_data_digits():
    X, y = load_digits(return_X_y=True)

    return X, y
