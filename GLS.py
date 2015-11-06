# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:58:37 2015

@author: Haolin
"""

import numpy as np
import Util
import GEVFunc

class GLS(object):
    def __init__(self, xi=-0.2567, reg=0., iterations=100, tolerance=1e-5):
        self.xi = xi
        self.reg = reg
        self.iterations = iterations
        self.tol = tolerance
        self._beta = None

    def getXi(self):
        return self.xi

    def setXi(self, xi):
        self.xi = xi

    def getRegular(self):
        return self.reg

    def setRegular(self, regular):
        self.reg = regular
        
    def getIterations(self):
        return self.iterations

    def setIterations(self, iterations):
        self.iterations = iterations

    def getTolerance(self):
        return self.tol

    def setTolerance(self, tol):
        self.tol = tol
        
    def getBeta(self):
        return self._beta

    def setBeta(self, beta):
        self._beta = beta
        
    def calculateL(self, xi):
        if xi < -1: raise Exception("No L exists when xi < -1")
        if xi == 0: return 1 / np.e
        if xi == -1: return 1. 
        v = (np.power(1./(xi + 1), xi) - 1)/xi
        return GEVFunc.derivInverseLink(xi, v)

    def fit(self, X, Y):
        n, d = X.shape[0], X.shape[1]
        Y = Y.reshape(-1, 1) 
        xi = self.xi
        L = self.calculateL(xi)
        secOrd = np.matrix((X.T.dot(X) * L + self.reg * np.eye(d)) / n).I
        self._beta = np.zeros(d).reshape(-1, 1)
        
        t = 0
        while t < self.iterations:
            v = X.dot(self._beta)
            GEVFunc.clip(self.xi, v)
            Y_hat = GEVFunc.inverseLink(xi, v)
            firOrd = (X.T.dot(Y_hat - Y) + self.reg * self._beta) / n     
            newBeta = self._beta - secOrd.dot(firOrd)
            error = np.abs(newBeta - self._beta).sum()
#            if t > self.iterations - 30:
#                print error
            if error < self.tol:
                self._beta = newBeta
                break
            self._beta = newBeta
            t += 1
        self._beta = np.array(self._beta).flatten()
#        print "Iteraions: ", t
    
    def predict(self, X):
        v = X.dot(self._beta)
        GEVFunc.clip(self.xi, v)
        return GEVFunc.inverseLink(self.xi, v)

if __name__ == "__main__": 
    fileName = "data/german.data"
    preproc = False
    evalFunc = Util.brierScore
    
#    validateX, validateY, \
#    trainX, trainY, \
#    testX, testY = Util.readData(fileName, True, preproc)
    clf = GLS()
#    xis = np.linspace(-1., 1.5, 26)
#    bestScore, bestXi = 1e10, None
#    score, xi = Util.crossValidate(clf, validateX, validateY, \
#                                        evalFunc, 5, "Xi", xis)
#    print "bestScore = ", score, "bestXi = ", xi
    xi = -0.5
    clf.setXi(xi)
    k = 10
    scoreList = []
    data = np.loadtxt(fname=fileName, delimiter=",")
    for _ in xrange(k):
        trainX, trainY, testX, testY = Util.readData(data, preproc=preproc)
        clf.fit(trainX, trainY)
        s = evalFunc(clf.predict(testX), testY)
        print "score = ", s
        scoreList.append(s)
    print "mean score = ", sum(scoreList)/k