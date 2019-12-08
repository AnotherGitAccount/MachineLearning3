# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import argparse
from contextlib import contextmanager


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from rdkit import Chem
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek


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


def display_confusion_matrix(model, X_train, y_train, X_test, y_test, save="confusion_matrix.pdf", title="Confusion Matrix"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("confusion_matrix:\n", conf_mat)

    labels = ["Class Inactive", "Class Active"]
    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt=".3f",
                linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title(title, size=12)
    plt.savefig(save)
    return

def get_info_features(y_LS, LS, save="Count.pdf"):
    unique, counts = np.unique(y_LS, return_counts=True)
    unique_y = dict(zip(unique, counts))
    print("Class 0: ", unique_y[0])
    print("Class 1: ", unique_y[1])
    print("Proportion: ", round(unique_y[0]/unique_y[1], 2), ": 1.0")
    # Plot the proportion
    count_before = sns.catplot(
        x="ACTIVE", kind="count", palette="ch:.25", data=LS)
    count_before.fig.suptitle(str(unique_y))
    count_before.savefig(save)

def get_RF_model():
    return RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=50, min_samples_leaf=1, bootstrap=True,
                                  max_features="log2", n_estimators=200, max_depth=1, criterion='gini', max_leaf_nodes=None)

def get_knn_model():
    return KNeighborsClassifier(n_neighbors=3)

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

    # -------------------------- Preprocessing techniques --------------------------- #

    # LEARNING
    # Create fingerprint features and output
    with measure_time("Creating fingerprint"):
        X_LS = create_fingerprints(LS["SMILES"].values)
        # drop duplicate
        data = pd.DataFrame(X_LS)
        data = data.drop_duplicates()
        X_LS = data.values

    # Drop also duplicate in the y_LS samples
    y_LS = LS["ACTIVE"].loc[data.index].values

    # Build the model (Random here -> 0.73015 on the platform)
    #model = get_RF_model()
    model = RandomForestClassifier(n_estimators=800, max_depth=700, n_jobs=-1)

    #Test with and without sampling
    X_train, X_test, y_train, y_test = train_test_split(X_LS, y_LS, test_size=0.25, random_state=1)  

    print("First tests without sampling:")
    #Without sampling
    get_info_features(y_train, pd.DataFrame({'ACTIVE':y_train}), save="Count_before.pdf")
    display_confusion_matrix(get_RF_model(), X_train, y_train, X_test, y_test, save="confusion_matrix_before.pdf", title="Confusion Matrix before sampling")

    print("\nSecond test with sampling:")
    #With sampling
    X, y = ADASYN().fit_sample(X_train, y_train)
    get_info_features(y, pd.DataFrame({'ACTIVE':y}), save="Count_after.pdf")
    display_confusion_matrix(get_RF_model(), X, y, X_test, y_test, save="confusion_matrix_after.pdf", title="Confusion Matrix after sampling")

    # ADD CODE HERE

    X, y = ADASYN().fit_sample(X_LS, y_LS)
    #do_cv_RF(X, y)
    #score = cross_val_score(model, X, y, cv=10, scoring="roc_auc")
    #print(np.mean(score))
    exit("No need to make submission now")
    with measure_time('Training'):
        model.fit(X, y)

    # PREDICTION
    TS = load_from_csv(args.ts)
    X_TS = create_fingerprints(TS["SMILES"].values)

    # Predict
    y_pred = model.predict_proba(X_TS)[:, 1]

    # Estimated AUC of the model
    auc_predicted = 0.75  # it seems a bit pessimistic, right?

    # Making the submission file
    fname = make_submission(y_pred, auc_predicted, 'test_over_sampling_DT')
    print('Submission file "{}" successfully written'.format(fname))