# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:00:57 2015

@author: Haolin
"""

import numpy as np
from GLS import GLS
import Util
from datetime import datetime as dt

def main_with_validation(fileName, preproc=False, evalFunc = Util.brierScore):
    trainX, trainY, testX, testY = Util.readData(fileName, preproc)
        
    clf = GLS()
    regs = np.logspace(-3, 3, 7)
    xis = np.linspace(-1., 1.6, 14)
    bestScore, bestXi, bestReg = 1e10, None, None

    for xi in xis:
        clf.setXi(xi)
        print "Current xi = ", xi
        score, reg = Util.crossValidate(clf, trainX, trainY, \
                                        evalFunc, 5, "Regular", regs)
        if score < bestScore:
            bestScore, bestXi, bestReg = score, xi, reg
    print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
    clf.setXi(bestXi)
    clf.setRegular(bestReg)
    clf.fit(trainX, trainY)
    testScore = evalFunc(clf.predict(testX), testY)
    with open("log/GLS_final_log.txt", 'a') as f:
        log = ','.join([dt.now().strftime("%Y/%m/%d %H:%M"), str(fileName), \
                        str(bestReg), str(bestXi), str(preproc), \
                        evalFunc.__name__, str(bestScore), str(testScore)])
        f.write(log + '\n')
    
if __name__ == "__main__":
    for name in ["vehicle", "german", "glass", "harberman", "pima", "letter-A"]:
        for p in [False, True]:
            for f in [Util.brierScore, Util.calibrationLoss]:
                main_with_validation("data/"+name+".data", p, f)
#    l = []    
#    with open("log/GLS_validation_log.txt") as f:
#        l = f.readlines()
#    l = [s.strip().split(',') for s in l]
#    for s in l:
#        main_prob(s[1], float(s[2]), float(s[3]), s[4] == "True", Util.__dict__[s[5]])
#    #main_label("data/letter-A.data",0.001,-0.5,False,Util.recall)