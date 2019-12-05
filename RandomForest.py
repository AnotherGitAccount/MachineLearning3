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
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem


# Build the model
model = RandomForestClassifier()

param_grid = {
    'min_samples_split': [3], 
    'n_estimators' : [300, 400, 1000],
    'max_depth': [100, 300],
    'max_features': [5, "log2", "sqrt"],
    'bootstrap' : [True, False]
}

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
        X[i,:] = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=124)

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
        handle.write('Chem_{:d},{}\n'.format(0,auc_predicted))

        for n,idx in enumerate(indexes):

            if np.isnan(y_predicted[n]):
                raise ValueError('The prediction cannot be NaN')
            line = 'Chem_{:d},{}\n'.format(idx, y_predicted[n])
            handle.write(line)
    return file_name

def grid_search_wrapper(X_LS, y_LS, X_test, y_test, refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(model, param_grid, refit=True, cv=skf, return_train_score=True, n_jobs=-1, scoring="roc_auc")
    grid_search.fit(X_LS, y_LS)
    print("lol")
    # make the predicctions
    y_pred = grid_search.predict_proba(X_test)
    print(y_pred.shape, y_test.shape)
    print("lol2")
    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    print(roc_auc_score(y_test, y_pred[:,1]))
    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    #print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

    return grid_search

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

    # -------------------------- Decision Tree --------------------------- #

    # LEARNING
    # Create fingerprint features and output
    with measure_time("Creating fingerprint"):
        X_LS = create_fingerprints(LS["SMILES"].values)
    y_LS = LS["ACTIVE"].values

    with measure_time('Training'):
        (X_train, X_test, y_train, y_test) = train_test_split(X_LS, y_LS, random_state=42, test_size=0.85, train_size=0.15)
        grid_search = grid_search_wrapper(X_train, y_train, X_test, y_test, refit_score='accuracy_score')
        results = pd.DataFrame(grid_search.cv_results_)
        #print(results)
        results = results.sort_values(by='mean_test_score', ascending=False)
        print(results[['mean_test_score', 'mean_train_score',
                 'param_solver', 'param_activation', 'param_learning_rate',
                 'param_hidden_layer_sizes', 'param_alpha']].head())
        #model.fit(X_LS, y_LS)
    exit()
    # PREDICTION
    TS = load_from_csv(args.ts)
    X_TS = create_fingerprints(TS["SMILES"].values)

    # Predict
    y_pred = grid_search.predict_proba(X_TS)[:,1]

    # Estimated AUC of the model
    auc_predicted = 0.50 # it seems a bit pessimistic, right?

    # Making the submission file
    fname = make_submission(y_pred, auc_predicted, 'my_test')
    print('Submission file "{}" successfully written'.format(fname))
