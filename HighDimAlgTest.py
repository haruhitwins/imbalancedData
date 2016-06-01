# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 14:04:05 2015

@author: Haolin
"""

import numpy as np
from HDA import HighDimAlg
from GLS import GLS
#from GEVCanReg import GEVCanReg
import Util
from datetime import datetime as dt

def main_random(fileName, clf, p=5, k=3, preproc=False, evalFunc = Util.brierScore):
    data = np.loadtxt(fname=fileName, delimiter=",")
    hda = HighDimAlg(clf, "random", p, k)
    scoreList = []
    nRunning = 10
    for _ in range(nRunning):
        trainX, trainY, testX, testY = Util.readData(data, False, preproc)
        hda.fit(trainX, trainY)
        s = evalFunc(hda.predict(testX), testY)
        print("score = ", s)
        scoreList.append(s)
    score = sum(scoreList)/nRunning
    print("mean score = ", score)
    with open("log/HDA_test_log_random.txt", 'a') as f:
        log = ','.join([dt.now().strftime("%Y/%m/%d %H:%M"), str(fileName), \
                        "GLS", str(p), str(k), \
                        str(preproc), evalFunc.__name__, str(score)])
        f.write(log + '\n')

def main_ordinal(fileName, clf, k=3, preproc=False, evalFunc = Util.brierScore):
    data = np.loadtxt(fname=fileName, delimiter=",")
    hda = HighDimAlg(clf, "ordinal", kTimes=k)
    scoreList = []
    nRunning = 10
    for _ in range(nRunning):
        trainX, trainY, testX, testY = Util.readData(data, False, preproc)
        hda.fit(trainX, trainY)
        s = evalFunc(hda.predict(testX), testY)
        print("score = ", s)
        scoreList.append(s)
    score = sum(scoreList)/nRunning
    print("mean score = ", score)
    with open("log/HDA_test_log_ordinal.txt", 'a') as f:
        log = ','.join([dt.now().strftime("%Y/%m/%d %H:%M"), str(fileName), \
                        "GLS", str(k), \
                        str(preproc), evalFunc.__name__, str(score)])
        f.write(log + '\n')
        
if __name__ == "__main__":
#    l = []    
#    with open("log/GLS_validation_log.txt") as f:
#        l = f.readlines()
#    l = [s.strip().split(',') for s in l]
#    for s in l:
#        gls = GLS(xi=float(s[3]), reg=float(s[2]))
#        for k in [3,4,5]:
#            main_random(s[1], gls, 3, k, s[4] == "True", Util.__dict__[s[5]])
    l = []    
    with open("log/GLS_validation_log.txt") as f:
        l = f.readlines()
    l = [s.strip().split(',') for s in l]
    for s in l:
        if s[1] == "data/harberman.data": continue
        gls = GLS(xi=float(s[3]), reg=float(s[2]))
        for k in [3,4]:
            main_ordinal(s[1], gls, k, s[4] == "True", Util.__dict__[s[5]])