#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn_lvq import GlvqModel

import random
np.random.seed(42)
random.seed(42)

from ceml.sklearn import generate_counterfactual
from utils import covariance_to_correlation, load_data_iris, load_data_breast_cancer, load_data_wine, load_data_digits, get_delta_overlap
from models_mp import SeparatingHyperplane, GMLVQ


n_kfold_splits = 3

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <dataset_desc> <model_desc>")
    else:
        dataset_desc = sys.argv[1]
        model_desc = sys.argv[2]

        n_components = None
        if dataset_desc == "iris":
            X, y = load_data_iris()
        elif dataset_desc == "breastcancer":
            X, y = load_data_breast_cancer()
        elif dataset_desc == "wine":
            X, y = load_data_wine()
        elif dataset_desc == "digits":
            X, y = load_data_digits()
        labels = np.unique(y)
        n_dict_components = 10
        print(labels)
        print(f"Dimensionality: {X.shape[1]}")

        # Results:
        n_wrong_classification = 0
        n_not_found = 0
        corr_matrices = []
        original_samples = []
        counterfactuals = []
        counterfactuals_labels = []
        causal_counterfactuals = []
        counterfactual_causal_dist = []
        deltas = []
        causal_deltas = []
        deltas_overlap = []

        # k-fold cross validation
        kf = KFold(n_splits=n_kfold_splits, shuffle=True)
        for train_indices, test_indices in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_indices, :], X[test_indices], y[train_indices], y[test_indices]
            print(f"Train size: {X_train.shape}\nTest size: {X_test.shape}")

            # Dcitionary learning - smth. like "sparse coding"
            print("Learning dictionary")
            dict_learner = MiniBatchDictionaryLearning(n_components=n_dict_components, transform_algorithm='omp',n_jobs=-1, alpha=10., n_iter=1000)
            dict_learner.fit(X_train)
            X_train_coeff, X_test_coeff = dict_learner.transform(X_train), dict_learner.transform(X_test)
            dict_mat = dict_learner.components_
            X_train, X_test = dict_learner.transform(X_train) @ dict_learner.components_, dict_learner.transform(X_test) @ dict_learner.components_
            print("Dictinary learned")

            # Preprocessing
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train);X_test = scaler.transform(X_test);print("StandardScaler is applied to data.")

            # Fit and evaluate model
            model = None
            if model_desc == "logreg":
                model = LogisticRegression(multi_class='multinomial')
            elif model_desc == "glvq":
                model = GlvqModel(prototypes_per_class=3, max_iter=1000)

            cov = GraphicalLasso(alpha=.8).fit(X_train).covariance_ # Compute and turn covariance into correlation matrix
            ##cov = GraphicalLassoCV().fit(X_train).covariance_ # Compute and turn covariance into correlation matrix
            corr = covariance_to_correlation(cov)
            #corr = np.identity(X_train.shape[1])
            corr_matrices.append(corr)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
            print()

            # Compute counterfactual explanations of all test samples
            cf = None
            if model_desc == "logreg":
                cf = SeparatingHyperplane(model.coef_, model.intercept_, epsilon=1e-5)  # Object for computing causal counterfactuals
            elif model_desc == "gmlvq" or model_desc == "glvq":
                cf = GMLVQ(model, epsilon=1e-2)

            n_test = X_test.shape[0]
            for i in range(n_test):
                # Get current data point, its ground truth label and compute a random target label
                x_orig, y_orig = X_test[i,:],y_test[i]
                if model.predict([x_orig]) != y_orig:
                    n_wrong_classification += 1
                    continue

                y_target = random.choice(list(filter(lambda l: l != y_orig, labels)))
                I = np.identity(x_orig.shape[0])

                # Compute counterfactuals with and without causality constraint
                try:
                    opt = "mp"
                    regularizer="l1"  # L2 -> All features are changed, L1 -> Only very few features are changed (huge different for causal vs. non-causal counterfactual)

                    # Without causality constraint
                    opt_args = {"epsilon": 1.e-2, "solver": cp.MOSEK, "solver_verbosity": True, "max_iter": 100}
                    x_cf, delta = cf.compute_counterfactual(x_orig, y_target, corr=I, regularizer=regularizer)#, optimizer_args=opt_args)
                    if x_cf is None:
                        print("Computation of counterfactual failed!", y_target)
                        n_not_found += 1
                        continue
                    if model.predict([x_cf]) != y_target:
                        print("Wrong prediction on counterfactual")
                        continue
                    delta_cf = x_cf - x_orig

                    # With causality constraint
                    x_orig_prime = X_test_coeff[i,:] @ dict_mat    # Sparse coding via learned dictionary
                    corr_prime = dict_mat.T @ dict_mat_corr

                    x_cf2, delta2 = cf.compute_counterfactual(x_orig_prime, y_target, corr=corr_prime, regularizer=regularizer)
                    if x_cf2 is None:
                        print("Computation of causal counterfactual failed!")
                        n_not_found += 1
                        continue
                    if model.predict([x_cf2]) != y_target:
                        print("Wrong prediction on counterfactual")
                        #n_wrong_classification += 1
                        continue
                    delta_cf2 = x_cf2 - x_orig

                    # Evaluate closeness, number of non-zero changes, etc.
                    original_samples.append(x_orig)
                    counterfactuals_labels.append(y_target)
                    counterfactuals.append(x_cf)
                    causal_counterfactuals.append(x_cf2)
                    counterfactual_causal_dist.append(np.linalg.norm(x_cf - x_cf2, 1))

                    delta = np.round(delta, decimals=5) # Avoid very small number like 10^-10 which are likely a numerical artefact from optimization
                    delta2 = np.round(delta2, decimals=5)
                except Exception as ex:
                    print(ex, y_target)
                    n_not_found += 1

        # Compute final evaluation
        print("Final evaluation")
        print(f"Dim: {X.shape[1]}")
        print(f"Not found: {n_not_found}")
        print(f"Wong classification: {n_wrong_classification}")

        # Compute some statistics for each metric
        print(f"=>Difference:\nMean: {np.mean(counterfactual_causal_dist)}\nMedian: {np.median(counterfactual_causal_dist)}\nVar: {np.var(counterfactual_causal_dist)}\nStd: {np.std(counterfactual_causal_dist)}")
        print(f"=>Overlap:\nMean: {np.mean(deltas_overlap)}\nMedian: {np.median(deltas_overlap)}\nVar: {np.var(deltas_overlap)}\nStd: {np.std(deltas_overlap)}")
