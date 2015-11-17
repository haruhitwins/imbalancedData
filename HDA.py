# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 10:19:27 2015

@author: Haolin
"""
import numpy as np
from GLS import GLS

class HighDimAlg(object):
    def __init__(self, clf, scheme="random", patchSize=100, kTimes=10):        
        self.clf = clf
        self.setScheme(scheme)
        self.p = patchSize
        self.k = kTimes
        self.betas = []
        self.features = []
        self.algName = str(clf.__class__)
            
    def getScheme(self):
        return self.scheme
    
    def setScheme(self, scheme):
        scheme = scheme.lower()
        assert scheme == "random" or scheme == "ordinal", \
               "param 'scheme' must be 'random' or 'ordinal'"
        self.scheme = scheme
        
    def getPatchSize(self):
        return self.p
        
    def setPatchSize(self, size):
        self.p = size
        
    def getkTimes(self):
        return self.k
        
    def setkTimes(self, k):
        self.k = k
    
    def setXi(self, xi):
        self.clf.setXi(xi)
        
    def setRegular(self, reg):
        self.clf.setRegular(reg)
        
    def generateFeatures(self, X):
        d = X.shape[1]
        self.features = []
        if self.scheme == "random":
            assert d >= self.p, \
            "patchSize is too large. Only %d features available but given %d" \
            % (d, self.p)
            for _ in xrange(self.k):
                selected = np.array([False] * d)
                selected[:self.p] = True
                np.random.shuffle(selected)
                self.features.append(selected)
        else:
            assert d >= self.k * 2, "Partition number k is too large."
            ps = d/self.k
            for i in xrange(self.k-1):
                selected = np.array([False] * d)
                selected[i * ps : (i+1) * ps] = True
                self.features.append(selected)
            selected = np.array([False] * d)
            selected[(i+1) * ps :] = True
            self.features.append(selected)
        
    def fit(self, X, Y):
        n = X.shape[0]
        predictY = np.zeros(n)
        self.betas = []
        self.generateFeatures(X)
        for i in xrange(self.k):
            trainX = X[:, self.features[i]]
            self.clf.fit(trainX, Y - predictY)
            self.betas.append(self.clf.getBeta())
            predictY += self.clf.predict(trainX)
    
    def predict(self, X):
        n = X.shape[0]
        predictY = np.zeros(n)
        for i in xrange(self.k):
            testX = X[:, self.features[i]]
            self.clf.setBeta(self.betas[i])
            predictY += self.clf.predict(testX)            
        return predictY
        
if __name__ == "__main__":
    X = np.array([[1,2,3,4,5,6,7],\
                  [7,6,5,4,3,2,1]])
    Y = np.array([0, 1])              
    gls = GLS()
    hda = HighDimAlg(gls, scheme="random", patchSize=4, kTimes=3)
    hda.fit(X, Y)
    print hda.predict(X)