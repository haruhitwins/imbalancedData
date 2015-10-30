# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:34:00 2015

@author: Haolin
"""

import numpy as np
from GEVCanReg import GEVCanReg
import Util
from datetime import datetime as dt

def main_with_validation(fileName, preproc=False, evalFunc = Util.brierScore):
    validateX, validateY, \
    trainX, trainY, \
    testX, testY = Util.readData(fileName, True, preproc)
        
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
    print "bestReg, bestXi = ", bestReg, bestXi
    with open("log/GEV_validation_log.txt", 'a') as f:
        log = ','.join([dt.now().strftime("%Y/%m/%d %H:%M"), str(fileName), \
                        str(bestReg), str(bestXi), str(preproc), evalFunc.__name__])
        f.write(log + '\n')

def main_prob(fileName, reg, xi, preproc=False, evalFunc = Util.brierScore):
    data = np.loadtxt(fname=fileName, delimiter=",")
    scoreList = []
    clf = GEVCanReg()
    clf.setRegular(reg)
    clf.setXi(xi)
    k = 10
    
    for _ in xrange(k):
        trainX, trainY, testX, testY = Util.readData(data, False, preproc)
        clf.fit(trainX, trainY)
        s = evalFunc(clf.predict(testX), testY)
        print "score = ", s
        scoreList.append(s)
    score = sum(scoreList)/k
    print "mean score = ", score
    with open("log/GEV_test_log.txt", 'a') as f:
        log = ','.join([dt.now().strftime("%Y/%m/%d %H:%M"), str(fileName), \
                        str(preproc), evalFunc.__name__, str(score)])
        f.write(log + '\n')

def main_label(fileName, reg, xi, preproc=False, evalFunc = Util.f1):
    data = np.loadtxt(fname=fileName, delimiter=",")
    scoreList = []
    clf = GEVCanReg()
    clf.setRegular(reg)
    clf.setXi(xi)
    k = 10
    
    for _ in xrange(k):
        trainX, trainY, testX, testY = Util.readData(data, False, preproc)
        clf.fit(trainX, trainY)
        s = evalFunc(Util.probToLabel(clf.predict(testX)), testY)
        print "score = ", s
        scoreList.append(s)
    print "mean score = ", sum(scoreList)/k
    
if __name__ == "__main__":
#    for name in ["vehicle", "german", "glass", "harberman", "pima", "letter-A"]:
#        for p in [False, True]:
#            for f in [Util.brierScore, Util.calibrationLoss]:
#                main_with_validation("data/"+name+".data", p, f)
    l = []    
    with open("log/GEV_validation_log.txt") as f:
        l = f.readlines()
    l = [s.strip().split(',') for s in l]
    for s in l:
        main_prob(s[1], float(s[2]), float(s[3]), s[4] == "True", Util.__dict__[s[5]])
    #main_label("data/letter-A.data",0.001,-0.5,False,Util.recall)