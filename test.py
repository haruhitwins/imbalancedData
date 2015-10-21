# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:34:00 2015

@author: Haolin
"""

import numpy as np
from GEVCanReg import GEVCanReg
import Util

def main_with_validation():
    data = np.loadtxt(fname="data/letter-A.data", delimiter=",")
    n = data.shape[0]

    np.random.shuffle(data)
    train = data[:n * 0.7]
    test = data[n * 0.7:]
    validate = train[:n * 0.3]

    validateX = validate[:, 1:]
    validateY = validate[:, 0].flatten()
    trainX = train[:, 1:]
    trainY = train[:, 0].flatten()
    testX = test[:, 1:]
    testY = test[:, 0].flatten()

    clf = GEVCanReg()
    regs = np.logspace(-3, 3, 7)
    xis = np.linspace(-1., 1.5, 26)
    bestScore, bestXi, bestReg = 1e10, None, None

    for xi in xis:
        clf.setXi(xi)
        score, reg = Util.crossValidate(clf, validateX, validateY, \
                                        Util.brierScore, 5, "Regular", regs)
        if score < bestScore:
            bestScore, bestXi, bestReg = score, xi, reg
            print "bestScore, bestXi, bestReg = ", score, xi, reg
    print "bestReg, bestXi = ", bestReg, bestXi #0.001, -0.5
    clf.setRegular(bestReg)
    clf.setXi(bestXi)
    print "fitting training data"
    clf.fit(trainX, trainY)
    print "brierScore = ", Util.brierScore(clf.predict(testX), testY)

def main():
    data = np.loadtxt(fname="data/letter-A.data", delimiter=",")
    n = data.shape[0]
    scoreList = []
    clf = GEVCanReg()
    clf.setRegular(0.001)
    clf.setXi(-0.5)
    k = 10
    
    for _ in xrange(k):
        np.random.shuffle(data)
        train = data[:n * 0.7]
        test = data[n * 0.7:]
        trainX = train[:, 1:]
        trainY = train[:, 0].flatten()
        testX = test[:, 1:]
        testY = test[:, 0].flatten()
        clf.fit(trainX, trainY)
        s = Util.brierScore(clf.predict(testX), testY)
        print "brierScore = ", s
        scoreList.append(s)
    print "mean score = ", sum(scoreList)/k
    
if __name__ == "__main__":
    main()
