# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:00:57 2015

@author: Haolin
"""

import numpy as np
from GLS import GLS
import Util
from datetime import datetime as dt

def main_with_validation(fileName, intercept=False, evalFunc = Util.brierScore):
    trainX, trainY, testX, testY = Util.readData(fileName, intercept=intercept)
        
    clf = GLS()
    regs = np.logspace(-3, 3, 7)
    xis = np.linspace(-1., 1.5, 26)
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
                        str(bestReg), str(bestXi), str(intercept), \
                        evalFunc.__name__, str(bestScore), str(testScore)])
        f.write(log + '\n')
    
if __name__ == "__main__":
    for name in ["trans_cleveland-0_vs_4", \
                 "trans_dermatology-6", \
                 "trans_lymphography-normal-fibrosis", \
                 "trans_page-blocks-1-3_vs_4", \
                 "trans_segment0", \
                 "trans_vowel0", \
                 "trans_zoo-3"]:
        for p in [False, True]:
            for f in [Util.brierScore, Util.calibrationLoss]:
                main_with_validation("data/"+name+".data", p, f)
