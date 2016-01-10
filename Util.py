# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:22:35 2015

@author: Haolin
"""

import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

def brierScore(predictY, trueY):
#    y = trueY.copy()
#    y[y != 1] = 0
#    return np.square(predictY - y).mean()
    return brier_score_loss(trueY, predictY)

def stratifiedBrierScore(predictY, trueY):
    y = trueY.copy()
    y[y != 1] = 0
    pos, neg = y == 1, y == 0
    return np.square(predictY[pos] - y[pos]).mean(), np.square(predictY[neg] - y[neg]).mean()
    
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

def auc(predictY, trueY):
    return roc_auc_score(trueY, predictY)

def matthews(predictY, trueY):
    return matthews_corrcoef(trueY, predictY)    

def gmean(predictY, trueY):
    cm = confusion_matrix(trueY, predictY).astype(float)
    return np.sqrt(cm[0,0] / (cm[0,0] + cm[0,1]) * cm[1,1] / (cm[1,1] + cm[1,0]))
    
def accuracy(predictY, trueY):
    return accuracy_score(trueY, predictY)

def evaluate(predY, testY):
    b = brierScore(predY, testY)
    b1, b0 = stratifiedBrierScore(predY, testY)
    c = calibrationLoss(predY, testY)
    a = auc(predY, testY)
    predY = probToLabel(predY)
    r = recall(predY, testY)
    p = precision(predY, testY)
    f = f1(predY, testY)
    m = matthews(predY, testY)
    g = gmean(predY, testY)
    return b, b1, b0, c, a, r, p, f, m, g
    
def probToLabel(y, threshold = 0.5):
    res = np.zeros(len(y))
    res[y >= threshold] = 1
    return res
    
def crossValidate(classifier, X, Y, evalFunc, k, name, params):
    skf = cross_validation.StratifiedKFold(Y, n_folds = k)
    setFunc = classifier.__getattribute__("set" + name)
    bestScore, bestParam = 1e10, None
    
    for param in params:
        setFunc(param)
        scores = []
        for train_id, test_id in skf:
            classifier.fit(X[train_id], Y[train_id])
            predictY = classifier.predict(X[test_id])
#            if not (evalFunc == brierScore or evalFunc == calibrationLoss):
#                predictY = probToLabel(predictY)
            try:
                scores.append(evalFunc(predictY, Y[test_id]))
            except:
                continue
        score = np.array(scores).mean()
        if score < bestScore:
            bestScore = score
            bestParam = param
            #print "bestScore = ", score, "with param = ", param
    return (bestScore, bestParam)

def standardization(X):
    return preprocessing.scale(X)
    
def readData(source, preproc=False, intercept=True, testSize=0.3):
    if type(source) == str:
        data = np.loadtxt(fname=source, delimiter=",")
    else:
        data = source
    y = data[:, 0]
    pos = data[y == 1, :]
    neg = data[y != 1, :]
    neg[:, 0] = 0
    del y, data
    
    pn, nn = pos.shape[0], neg.shape[0]
    if testSize != 0:
        assert nn * testSize >= 1
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    test = np.vstack((pos[:pn * testSize], neg[:nn * testSize]))
    train = np.vstack((pos[pn * testSize:], neg[nn * testSize:]))
    del pos, neg
    
    trainX, trainY = train[:, 1:], train[:, 0]
    testX, testY = test[:, 1:], test[:, 0]

    if preproc: 
        trainX = standardization(trainX)
        testX = standardization(testX)

    if intercept:
        n = trainX.shape[0]
        trainX = np.hstack((trainX, np.ones(n).reshape(n, 1)))
        n = testX.shape[0]
        testX = np.hstack((testX, np.ones(n).reshape(n, 1)))
    return trainX, trainY, testX, testY

def calculateP(source):
    data = np.loadtxt(fname=source, delimiter=",")
    n = data.shape[0]
    return (data[:, 0] == 1).sum() / float(n)

def oneVsAll(X, clfs):
    predYs = np.zeros(len(clfs) * X.shape[0]).reshape(len(clfs), X.shape[0])
    for i, clf in enumerate(clfs):
        predYs[i] = clf.predict(X)
    return predYs.argmax(0)


       
if __name__ == "__main__":
    print "This is Util module."