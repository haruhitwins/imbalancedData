# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:22:35 2015

@author: Haolin
"""

import numpy as np
from sklearn import cross_validation

def brierScore(preditY, trueY):
    y = trueY.copy()
    y[y != 1] = 0
    return np.square(preditY - y).mean()
    
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
    
if __name__ == "__main__":
    print "This is Util module."