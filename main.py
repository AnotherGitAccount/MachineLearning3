# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import utils
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from rdkit import Chem
from rdkit.Chem import AllChem


def doDecisionTree(X, y):
    depths = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500]
    
    scores = []
    for depth in depths:
        classifier = DecisionTreeClassifier(max_depth=depth)
        score = cross_val_score(classifier, X, y, cv=10, scoring="roc_auc")
        scores.append(np.mean(score))
    
    return depths, scores


def doKNN(X, y):
    ks = [5, 10, 20, 50, 100, 500]
    
    scores = []
    for k in ks:
        classifier = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(classifier, X, y, cv=10, scoring="roc_auc")
        scores.append(np.mean(score))
        
    return ks, scores

    
def doMLP(X, y):
    print(X.shape)
    layers = [5]#[1, 2, 3, 4, 5, 10, 20, 30]   #[20, 30, 40, 50, 60]
    neurones = [124]# [1, 2, 3, 4, 5, 10, 20, 30] #[20, 30, 40, 50, 60] 
    
    scores = []
    for n in neurones:
        n_score = []
        for l in layers:
            dims = [n for _ in range(l)]
            classifier = MLPClassifier(hidden_layer_sizes=dims)
            score = cross_val_score(classifier, X, y, cv=10, scoring="roc_auc")
            n_score.append(np.mean(score))
        scores.append(n_score)
    return layers, neurones, scores
    
def doRandomForest(X, y):
    ts = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    depths = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    scores = []
    for t in ts:
        t_score = []
        for depth in depths:
            classifier = RandomForestClassifier(n_estimators=t, max_depth=depth)
            score = cross_val_score(classifier, X, y, cv=3, scoring="roc_auc")
            t_score.append(np.mean(score))
        scores.append(t_score)
    return ts, depth, scores
    
           
if __name__ == "__main__":

    METHOD = "RF"
   
    LS = utils.load_from_csv("data/training_set.csv")
    TS = utils.load_from_csv("data/test_set.csv")
    
    X_LS = utils.create_fingerprints(LS["SMILES"].values)
    Y_LS = LS["ACTIVE"].values
        
    X_TS = utils.create_fingerprints(TS["SMILES"].values)
    
    if METHOD == "DT":
        depths, scores = doDecisionTree(X_LS, Y_LS)
        print(scores)

    elif METHOD == "KNN":
        depths, scores = doKNN(X_LS, Y_LS)
        print(scores)
   
        classifier_knn = KNeighborsClassifier(n_neighbors=50)
        classifier_knn.fit(X_LS, Y_LS)
        pred = classifier_knn.predict_proba(X_TS)
        auc_predicted = 0.7
        fname = utils.make_submission(pred[:,1], auc_predicted, 'knn_50')
        print('Submission file "{}" successfully written'.format(fname))

    elif METHOD == "RF":
        #ts, depths, scores = doRandomForest(X_LS, Y_LS)
        #print(scores)
        
        classifier_rf = RandomForestClassifier(n_estimators=800, max_depth=700)
        classifier_rf.fit(X_LS, Y_LS)
        pred = classifier_rf.predict_proba(X_TS)
        auc_predicted = 0.78
        fname = utils.make_submission(pred[:,1], auc_predicted, 'final')
        print('Submission file "{}" successfully written'.format(fname))
        
    elif METHOD == "MLP":
        layers, neurones, scores = doMLP(X_LS, Y_LS)
        print(scores)