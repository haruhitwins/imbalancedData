# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:34:00 2015

@author: Haolin
"""

import numpy as np
from GEVCanReg import GEVCanReg
import Util

data = np.loadtxt(fname = "letter-A.data", delimiter = ",")
n = data.shape[0]
np.random.shuffle(data)
train = data[:n * 0.7]
test = data[n * 0.7:]
validate = train[:n * 0.3]
#==============================================================================
# X = np.array([[1,2,3], [3,2,1], [1,1,1], [2,3,3]])
# Y = np.array([1,-1,-1,1])
#==============================================================================
validateX = validate[:,1:]
validateY = validate[:,0].flatten()
trainX = train[:,1:]
trainY = train[:,0].flatten()
testX = test[:,1:]
testY = test[:,0].flatten()


regs = np.logspace(-3,3,7)
xis = np.linspace(-1,1.5,26)
bestScore, bestXi, bestReg = 1e10, None, None

clf = GEVCanReg()
for xi in xis:
    clf.setXi(xi)
    score, reg = Util.crossValidate(clf, validateX, validateY, Util.brierScore, 5, "Regular", regs)
    if score < bestScore:
        bestScore, bestXi, bestReg = score, xi, reg
        print "bestScore, bestXi, bestReg = ", score, xi, reg
clf.setRegular(bestReg)
clf.setXi(bestXi)
print "begin to fit training data"
clf.fit(trainX, trainY)
print "begin to predict"
print "brierScore = ", Util.brierScore(clf.predict(testX), testY)