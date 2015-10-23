# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:34:00 2015

@author: Haolin
"""

import numpy as np
from GEVCanReg import GEVCanReg
import Util

def main_with_validation(fileName, preproc=False, evalFunc = Util.brierScore):
    data = np.loadtxt(fname=fileName, delimiter=",")
    if preproc: data = Util.standardization(data)
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
        print "Current xi = ", xi
        score, reg = Util.crossValidate(clf, validateX, validateY, \
                                        evalFunc, 5, "Regular", regs)
        if score < bestScore:
            bestScore, bestXi, bestReg = score, xi, reg
            print "bestScore, bestXi, bestReg = ", score, xi, reg
    print "bestReg, bestXi = ", bestReg, bestXi
    clf.setRegular(bestReg)
    clf.setXi(bestXi)
    print "fitting training data"
    clf.fit(trainX, trainY)
    print "score = ", evalFunc(clf.predict(testX), testY)

def main_prob(fileName, reg, xi, preproc=False, evalFunc = Util.brierScore):
    data = np.loadtxt(fname=fileName, delimiter=",")
    if preproc: data = Util.standardization(data)
    n = data.shape[0]
    scoreList = []
    clf = GEVCanReg()
    clf.setRegular(reg)
    clf.setXi(xi)
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
        s = evalFunc(clf.predict(testX), testY)
        print "score = ", s
        scoreList.append(s)
    print "mean score = ", sum(scoreList)/k

def main_label(fileName, reg, xi, preproc=False, evalFunc = Util.f1):
    data = np.loadtxt(fname=fileName, delimiter=",")
    if preproc: data = Util.standardization(data)
    n = data.shape[0]
    scoreList = []
    clf = GEVCanReg()
    clf.setRegular(reg)
    clf.setXi(xi)
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
        s = evalFunc(Util.probToLabel(clf.predict(testX)), testY)
        print "score = ", s
        scoreList.append(s)
    print "mean score = ", sum(scoreList)/k
    
if __name__ == "__main__":
    #main_with_validation("data/vehicle.data",preproc=True)
    #main_prob("data/letter-A.data",0.001,-0.5,False,Util.brierScore)
    main_label("data/letter-A.data",0.001,-0.5,False,Util.recall)