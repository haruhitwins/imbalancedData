# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:58:37 2015

@author: Haolin
"""

import numpy as np
import Util
import GEVFunc

class GLS(object):
    def __init__(self, xi=-0.2567, reg=0., iterations=300, tolerance=1e-10):
        self.xi = xi
        self.reg = reg
        self.iterations = iterations
        self.tol = tolerance
        self._beta = None
        self.step = 1.0

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
        #secOrd = np.linalg.pinv((X.T.dot(X) * L + self.reg * np.eye(d)) / n)
#        I = np.eye(d)
#        I[-1, -1] = 0 #Not regularize the intercept
        secOrd = np.linalg.pinv(X.T.dot(X) * L / n + self.reg * np.eye(d))
#        if not type(self._beta) is np.ndarray:
        self._beta = np.zeros(d).reshape(-1, 1)
        self._beta[-1] = np.log(-np.log(Y.mean()))
        
        t = 0.
        while t < self.iterations:
            v = X.dot(self._beta)
            GEVFunc.clip(self.xi, v)
            Y_hat = GEVFunc.inverseLink(xi, v)
            #firOrd = (X.T.dot(Y_hat - Y) + self.reg * self._beta) / n  
#            tmp = self.reg * self._beta
#            tmp[-1] = 0
            firOrd = X.T.dot(Y_hat - Y) / n + self.reg * self._beta
            newBeta = self._beta - self.step * secOrd.dot(firOrd)
            error = np.abs(newBeta - self._beta).sum()
            if error < self.tol:
                self._beta = newBeta
                #print "BREAK!!!!, t = ", t
                break            
            self._beta = newBeta
            t += 1
        self._beta = np.array(self._beta).flatten()
    
    def predict(self, X):
        v = X.dot(self._beta)
        GEVFunc.clip(self.xi, v)
        return GEVFunc.inverseLink(self.xi, v)

if __name__ == "__main__": 
    regs = np.logspace(-4, 4, 10)
    xis = np.linspace(-1, 2, 31)    
    scores = np.zeros(100).reshape(10,10)

    for i in xrange(10):
        trainX, trainY, testX, testY = Util.readData("data/harberman.data")
        clf = GLS()
        bestScore, bestXi, bestReg = 1e10, None, None
        for xi in xis:
            clf.setXi(xi)
            score, reg = Util.crossValidate(clf, trainX, trainY, \
                                            Util.brierScore, 5, "Regular", regs)
            if score < bestScore:
                bestScore, bestXi, bestReg = score, xi, reg
        print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
        clf.setXi(bestXi)
        clf.setRegular(bestReg)
        clf.fit(trainX, trainY)
        scores[i] = Util.evaluate(clf.predict(testX).flatten(), testY)
    
    print scores.mean(axis=0)