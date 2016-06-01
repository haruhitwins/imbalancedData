# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:02:36 2015

@author: Haolin
"""

import numpy as np
from GLS import GLS
import Util
from datetime import datetime as dt

def main_with_validation(fileName, preproc=False, evalFunc = Util.brierScore):
    validateX, validateY, \
    trainX, trainY, \
    testX, testY = Util.readData(fileName, True, preproc)
        
    clf = GLS()
    xis = np.linspace(-1., 1.5, 26)
    bestScore, bestXi = Util.crossValidate(clf, validateX, validateY, \
                                        evalFunc, 5, "Xi", xis)
    print("bestScore, bestXi = ", bestScore, bestXi)
    with open("log/GLS_validation_log.txt", 'a') as f:
        log = ','.join([dt.now().strftime("%Y/%m/%d %H:%M"), str(fileName), \
                        str(bestXi), str(preproc), evalFunc.__name__])
        f.write(log + '\n')

def main_prob(fileName, xi, preproc=False, evalFunc = Util.brierScore):
    data = np.loadtxt(fname=fileName, delimiter=",")
    scoreList = []
    clf = GLS()
    clf.setXi(xi)
    k = 10
    
    for _ in range(k):
        trainX, trainY, testX, testY = Util.readData(data, False, preproc)
        clf.fit(trainX, trainY)
        s = evalFunc(clf.predict(testX), testY)
        print("score = ", s)
        scoreList.append(s)
    score = sum(scoreList)/k
    print("mean score = ", score)
    with open("log/GLS_test_log.txt", 'a') as f:
        log = ','.join([dt.now().strftime("%Y/%m/%d %H:%M"), str(fileName), \
                        str(preproc), evalFunc.__name__, str(score)])
        f.write(log + '\n')

def main_label(fileName, xi, preproc=False, evalFunc = Util.f1):
    data = np.loadtxt(fname=fileName, delimiter=",")
    scoreList = []
    clf = GLS()
    clf.setXi(xi)
    k = 10
    
    for _ in range(k):
        trainX, trainY, testX, testY = Util.readData(data, False, preproc)
        clf.fit(trainX, trainY)
        s = evalFunc(Util.probToLabel(clf.predict(testX)), testY)
        print("score = ", s)
        scoreList.append(s)
    print("mean score = ", sum(scoreList)/k)
    
if __name__ == "__main__":
#    for name in ["vehicle", "german", "glass", "harberman", "pima", "letter-A"]:
#        for p in [False, True]:
#            for f in [Util.brierScore, Util.calibrationLoss]:
#                main_with_validation("data/"+name+".data", p, f)
    l = []    
    with open("log/GLS_validation_log.txt") as f:
        l = f.readlines()
    l = [s.strip().split(',') for s in l]
    for s in l:
        main_prob(s[1], float(s[2]), s[3] == "True", Util.__dict__[s[4]])
    #main_label("data/letter-A.data",0.001,-0.5,False,Util.recall)