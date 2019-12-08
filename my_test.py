# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import argparse
from contextlib import contextmanager


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from rdkit import Chem
from rdkit.Chem import AllChem


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    print("{}...".format(label))
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError("File '{}' does not exists.".format(path))
    return pd.read_csv(path, delimiter=delimiter)


def create_fingerprints(chemical_compounds):
    """
    Create a learning matrix `X` with (Morgan) fingerprints
    from the `chemical_compounds` molecular structures.

    Parameters
    ----------
    chemical_compounds: array [n_chem, 1] or list [n_chem,]
        chemical_compounds[i] is a string describing the ith chemical
        compound.

    Return
    ------
    X: array [n_chem, 124]
        Generated (Morgan) fingerprints for each chemical compound, which
        represent presence or absence of substructures.
    """
    n_chem = chemical_compounds.shape[0]

    nBits = 124
    X = np.zeros((n_chem, nBits))

    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        X[i, :] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=124)

    return X


def make_submission(y_predicted, auc_predicted, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predicted: array [n_predictions, 1]
        if `y_predict[i]` is the prediction
        for chemical compound `i` (or indexes[i] if given).
    auc_predicted: float [1]
        The estimated ROCAUC of y_predicted.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Creating default indexes if not given
    if indexes is None:
        indexes = np.arange(len(y_predicted))+1

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"Chem_ID","Prediction"\n')
        handle.write('Chem_{:d},{}\n'.format(0, auc_predicted))

        for n, idx in enumerate(indexes):

            if np.isnan(y_predicted[n]):
                raise ValueError('The prediction cannot be NaN')
            line = 'Chem_{:d},{}\n'.format(idx, y_predicted[n])
            handle.write(line)
    return file_name

def doKNN(X_LS, Y_LS):
    #BEST = 52 d'après CV 10 fold

    neighbors = [i for i in range(50, 57, 1)]
    scores = []

    for k in neighbors:
        print(k)
        estimator = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(estimator, X_LS, Y_LS, cv=10, scoring="roc_auc")
        scores.append(np.mean(score))

    plt.figure()
    plt.plot(neighbors, scores, label="CV score")
    plt.xlabel("Complexity of the Model")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("KNN_scores.pdf")
    return

def doBaggingKnn(X_LS, Y_LS):
    #BEST = 52 d'après CV 10 fold
    neighbors = [i for i in range(50, 57, 1)]
    scores = []

    estimator = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=52))
    score = cross_val_score(estimator, X_LS, Y_LS, cv=10, scoring="roc_auc")
    scores.append(np.mean(score))

    print(scores)
    plt.figure()
    plt.plot([52], scores, label="CV score")
    plt.xlabel("Complexity of the Model")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("Bagging_KNN_CV.pdf")
    return

def doExtraTree(X_LS, Y_LS):
    #max_depth = 5 
    max_depths = [(i+1) for i in range(0, 35)]
    scores = []

    for depth in max_depths:
        print(depth)
        estimator = BaggingClassifier(base_estimator=ExtraTreeClassifier(max_depth=depth), n_estimators=100)
        score = cross_val_score(estimator, X_LS, Y_LS, cv=10, scoring="roc_auc")
        scores.append(np.mean(score))

    plt.figure()
    plt.plot(max_depths, scores, label="CV score")
    plt.xlabel("Complexity of the Model")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("extraTree_scores.pdf")
    return

def doDecisionTree(X_LS, Y_LS):
    max_depths = [50, 75, 100, 150, 200 , 300]
    scores = []
    for depth in max_depths:
        print(depth)
        estimator = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15), n_estimators=depth)
        score = cross_val_score(estimator, X_LS, Y_LS, cv=10, scoring="roc_auc")
        scores.append(np.mean(score))

    plt.figure()
    plt.plot(max_depths, scores, label="CV score")
    plt.xlabel("n_estimators of AdaBoostClassifier")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("DecisionTree_scores.pdf")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make a toy submission")
    parser.add_argument("--ls", default="data/training_set.csv",
                        help="Path to the learning set as CSV file")
    parser.add_argument("--ts", default="data/test_set.csv",
                        help="Path to the test set as CSV file")

    args = parser.parse_args()

    # Load training data
    LS = load_from_csv(args.ls)
    # Load test data
    TS = load_from_csv(args.ts)

    method = "RF"

    if method is "KNN":
        # -------------------------- Decision Tree --------------------------- #
        # LEARNING
        # Create fingerprint features and output
        print("KNN CLASSIFIER")
        with measure_time("Creating fingerprint"):
            X_LS = create_fingerprints(LS["SMILES"].values)
        y_LS = LS["ACTIVE"].values

        # Build the model
        model = KNeighborsClassifier(n_neighbors=52)

        with measure_time('Training'):
            #doKNN(X_LS, y_LS)
            model.fit(X_LS, y_LS)
        # PREDICTION
        TS = load_from_csv(args.ts)
        X_TS = create_fingerprints(TS["SMILES"].values)

        # Predict
        y_pred = model.predict_proba(X_TS)[:, 1]

        # Estimated AUC of the model
        auc_predicted = 0.712  # it seems a bit pessimistic, right?

        # Making the submission file
        fname = make_submission(y_pred, auc_predicted,
                                'KNN_52')
        print('Submission file "{}" successfully written'.format(fname))
    elif method is "BKNN":
        # -------------------------- Decision Tree --------------------------- #
        # LEARNING
        # Create fingerprint features and output
        print("BAGGING CLASSIFIER\n")
        with measure_time("Creating fingerprint"):
            X_LS = create_fingerprints(LS["SMILES"].values)
        y_LS = LS["ACTIVE"].values

        # Build the model
        model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=52))

        with measure_time('Training'):
            #doBaggingKnn(X_LS, y_LS)
            model.fit(X_LS, y_LS)
        # PREDICTION
        TS = load_from_csv(args.ts)
        X_TS = create_fingerprints(TS["SMILES"].values)

        # Predict
        y_pred = model.predict_proba(X_TS)[:, 1]

        # Estimated AUC of the model
        auc_predicted = 0.712  # it seems a bit pessimistic, right?

        # Making the submission file
        fname = make_submission(y_pred, auc_predicted,
                                'Bagging_KNN_useless')
        print('Submission file "{}" successfully written'.format(fname))
    elif method is "DT":
        # -------------------------- Decision Tree --------------------------- #
        # LEARNING
        # Create fingerprint features and output
        print("DT CLASSIFIER\n")
        with measure_time("Creating fingerprint"):
            X_LS = create_fingerprints(LS["SMILES"].values)
        y_LS = LS["ACTIVE"].values

        # Build the model
        model =  GradientBoostingClassifier(n_estimators=180)

        with measure_time('Training'):
            #doDecisionTree(X_LS, y_LS)
            #estimator = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15), n_estimators=200)
            score = cross_val_score(model, X_LS, y_LS, cv=10, scoring="roc_auc")
            print(np.mean(score))
            model.fit(X_LS, y_LS)
        # PREDICTION
        TS = load_from_csv(args.ts)
        X_TS = create_fingerprints(TS["SMILES"].values)

        # Predict
        y_pred = model.predict_proba(X_TS)[:, 1]

        # Estimated AUC of the model
        auc_predicted = 0.712  # it seems a bit pessimistic, right?

        # Making the submission file
        fname = make_submission(y_pred, auc_predicted,
                                'Bagging_KNN_useless')
        print('Submission file "{}" successfully written'.format(fname))
    elif method is "RF":
        # -------------------------- Decision Tree --------------------------- #
        # LEARNING
        # Create fingerprint features and output
        print("RandomForestClassifier\n")
        with measure_time("Creating fingerprint"):
            X_LS = create_fingerprints(LS["SMILES"].values)
        y_LS = LS["ACTIVE"].values

        # Build the model
        model = AdaBoostClassifier(n_estimators=50, base_estimator=SVC(probability=True, gamma='scale'))
        #model = SVC(kernel='sigmoid', probability=True, gamma='')

        with measure_time('Training'):
            #doDecisionTree(X_LS, y_LS)
            #estimator = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15), n_estimators=200)
            score = cross_val_score(model, X_LS, y_LS, cv=10, scoring="roc_auc")
            print(np.mean(score))
            model.fit(X_LS, y_LS)
        # PREDICTION
        TS = load_from_csv(args.ts)
        X_TS = create_fingerprints(TS["SMILES"].values)

        # Predict
        y_pred = model.predict_proba(X_TS)[:, 1]

        # Estimated AUC of the model
        auc_predicted = 0.712  # it seems a bit pessimistic, right?

        # Making the submission file
        fname = make_submission(y_pred, auc_predicted,
                                'RF')
        print('Submission file "{}" successfully written'.format(fname))
    else:
        print("Fail")
