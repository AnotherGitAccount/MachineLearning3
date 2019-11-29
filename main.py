# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import utils

if __name__ == "__main__":
    LS = utils.load_from_csv("data/training_set.csv")
    TS = utils.load_from_csv("data/test_set.csv")
    
    doDecisionTree(LS)
    doKNN(LS)
    doMLP(LS)
    doRandomForest(LS)

def doDecisionTree(learning_set):
    pass
    
def doKNN(learning_set):
    pass
    
def doMLP(learning_set):
    pass
    
def doRandomForest(learning_set):
    pass
