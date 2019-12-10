# ! /usr/bin/env python
# -*- coding: utf-8 -*-

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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingClassifier, VotingClassifier

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, EditedNearestNeighbours, CondensedNearestNeighbour
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.pipeline import make_pipeline


def get_RF_model(class_weight=None):
    return RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=50, min_samples_leaf=1, bootstrap=True,
                                  max_features="log2", n_estimators=1000, max_depth=None, criterion='gini', max_leaf_nodes=None, class_weight=class_weight)

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




def create_fingerprints(chemical_compounds, nBits=124):
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

    nBits = nBits
    X = np.zeros((n_chem, nBits))

    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        X[i,:] = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=nBits)

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

def plot_2d_space(X, y, label='Classes'): 
    plt.figure()  
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.savefig("PCA.pdf")

def display_confusion_matrix(model, X_train, y_train, X_test, y_test, save="confusion_matrix.pdf", title="Confusion Matrix"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("confusion_matrix:\n", conf_mat)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Score: ", roc_auc_score(y_true=y_test, y_score=y_pred))

    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt=".3f",
                linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title(title, size=10)
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


def do_sampling_research(X_train, y_train, X_test, y_test, nBits=124, info_features=False):

    print("First tests without sampling:")
    # Without sampling
    if info_features:
        get_info_features(y_train, pd.DataFrame({'ACTIVE': y_train}), save="Count_before_{}.pdf".format(nBits))
    pipeline = make_pipeline(ADASYN(sampling_strategy=0.25, random_state=64), BalancedRandomForestClassifier(n_estimators=1000, random_state=18, n_jobs=-1))
    display_confusion_matrix(pipeline, X_train, y_train, X_test, y_test,
                             save="confusion_matrix_before_{}.pdf".format(nBits), title="Confusion Matrix before sampling with {} nBits".format(nBits))

    print("\nSecond test with sampling:")
    # With sampling
    if info_features:
        get_info_features(y_train, pd.DataFrame({'ACTIVE': y_train}), save="Count_after_{}.pdf".format(nBits))
    pipeline = make_pipeline(ADASYN(sampling_strategy=0.35, random_state=64), BalancedRandomForestClassifier(n_estimators=1000, random_state=18, n_jobs=-1))
    display_confusion_matrix(pipeline, X_train, y_train, X_test, y_test,
                             save="confusion_matrix_after_{}.pdf".format(nBits), title="Confusion Matrix after sampling with {} nBits".format(nBits))


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
    nBits = [124, 250, 500, 1000, 1250, 1500, 1800]
    info_features = True
    for nbits in nBits:
        with measure_time("Creating fingerprint"):
            X_LS = create_fingerprints(LS["SMILES"].values, nBits=nbits)
            # drop duplicate
            data = pd.DataFrame(X_LS)
            data = data.drop_duplicates()
            X_LS = data.values

        # Drop also duplicate in the y_LS samples
        y_LS = LS["ACTIVE"].loc[data.index].values
        X_train, X_test, y_train, y_test = train_test_split(X_LS, y_LS, test_size=0.25, train_size=0.75, random_state=1)
        """
        select = SelectFromModel(estimator=BalancedRandomForestClassifier(n_estimators=500), threshold='mean')
        select = select.fit(X_train, y_train)
        X_train = select.transform(X_train)
        X_test = select.transform(X_test)
        """
        print(X_train.shape, X_test.shape)

        """
        data = pd.DataFrame(X_train)
        data['Target'] = y_train
        inactive_index = data[data['Target'] == 0].index
        length = len(inactive_index)
        drop_indices = np.random.choice(inactive_index, round(0.66*length), replace=False)
        data = data.drop(drop_indices)
        y_train = data.Target.values
        data = data.drop("Target", axis=1)
        X_train = data.values
        """
        print("nBits =",nbits)
        do_sampling_research(X_train, y_train, X_test, y_test, info_features=info_features, nBits=nbits)
        info_features = False
        pca = None

    exit("Preprocessing research end")
    # Build the model
    model = RandomForestClassifier()

    with measure_time('Training'):
        model.fit(X_LS, y_LS)

    # PREDICTION
    TS = load_from_csv(args.ts)
    X_TS = create_fingerprints(TS["SMILES"].values)

    # Predict
    y_pred = model.predict_proba(X_TS)[:,1]

    # Estimated AUC of the model
    auc_predicted = 0.7 # it seems a bit pessimistic, right?

    # Making the submission file
    fname = make_submission(y_pred, auc_predicted, 'toy_submission_DT')
    print('Submission file "{}" successfully written'.format(fname))






