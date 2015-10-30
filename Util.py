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
    bestScore, bestParam = 1e10, None
    
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
    
def readData(source, isValidate = False, preproc = False):
    if type(source) == str:
        data = np.loadtxt(fname=source, delimiter=",")
    else:
        data = source
    n = data.shape[0]

    np.random.shuffle(data)
    train, test = data[:n * 0.7], data[n * 0.7:]
    if isValidate:
        validate = train[:n * 0.3]
        validateX, validateY = validate[:, 1:], validate[:, 0].flatten()
        validateY[validateY != 1] = 0
    trainX, trainY = train[:, 1:], train[:, 0].flatten()
    trainY[trainY != 1] = 0
    testX, testY = test[:, 1:], test[:, 0].flatten()
    testY[testY != 1] = 0

    if preproc: 
        trainX = standardization(trainX)
        testX = standardization(testX)
        if isValidate : 
            validateX = standardization(validateX)
    
    if isValidate:
        return validateX, validateY, trainX, trainY, testX, testY
    else:
        return trainX, trainY, testX, testY

def calculateP(source):
    data = np.loadtxt(fname=source, delimiter=",")
    n = data.shape[0]
    return (data[:, 0] == 1).sum() / float(n)
        
if __name__ == "__main__":
    print "This is Util module."