# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:22:35 2015

@author: Haolin
"""

import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def brierScore(predictY, trueY):
    y = trueY.copy()
    y[y != 1] = 0
    return np.square(predictY - y).mean()
    
def calibrationLoss(predictY, trueY):
    y = trueY.copy()
    y[y != 1] = 0
    res = np.zeros(len(predictY))
    for i in xrange(10):
        bins = np.logical_and(i/10. < predictY, predictY <= (i+1)/10.)
        res[bins] = y[bins].sum() / float(bins.sum())
    return np.square(predictY - res).mean()

def recall(predictY, trueY):
    return recall_score(trueY, predictY)

def precision(predictY, trueY):
    return precision_score(trueY, predictY)
    
def f1(predictY, trueY):
    return f1_score(trueY, predictY)

def probToLabel(y, threshold = 0.5):
    res = np.zeros(len(y))
    res[y >= threshold] = 1
    return res
    
def crossValidate(classifier, X, Y, evalFunc, k, name, params):
    kf = cross_validation.KFold(X.shape[0], n_folds = k)
    setFunc = classifier.__getattribute__("set" + name)
    bestParam = None
    bestScore = 1e10
    
    for param in params:
        setFunc(param)
        scores = []
        for train_id, test_id in kf:
            classifier.fit(X[train_id], Y[train_id])
            predictY = classifier.predict(X[test_id])
            scores.append(evalFunc(predictY, Y[test_id]))
        score = sum(scores)/k
        if score < bestScore:
            bestScore = score
            bestParam = param
            #print "bestScore = ", score, "with param = ", param
    return (bestScore, bestParam)

def standardization(X):
    return preprocessing.scale(X)

if __name__ == "__main__":
    print "This is Util module."