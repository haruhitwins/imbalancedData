# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 10:19:27 2015

@author: Haolin
"""
import time
import numpy as np
from scipy.ndimage.filters import convolve
import Util
from GLS import GLS
from sklearn.datasets import fetch_mldata

class HighDimAlg(object):
    def __init__(self, clf, scheme="random", patchSize=100, kTimes=10):        
        self.clf = clf
        self.setScheme(scheme)
        self.p = patchSize
        self.k = kTimes
        self.betas = []
        self.features = []
        self.kernels = []
        self.width = 28
        self.algName = str(clf.__class__)
            
    def getScheme(self):
        return self.scheme
    
    def setScheme(self, scheme):
        scheme = scheme.lower()
        assert scheme == "random" or scheme == "ordinal" or scheme == "convolve", \
               "param 'scheme' must be 'random', 'ordinal' or 'convolve'"
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
    
    def getXi(self):
        return self.clf.getXi()
        
    def setRegular(self, reg):
        self.clf.setRegular(reg)
    
    def getRegular(self):
        return self.clf.getRegular()
        
    def setWidth(self, w):
        self.width = w
        
    def setKernels(self, k):
        self.kernels = k
        
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
        elif self.scheme == "ordinal":
            assert d >= self.k * 2, "Partition number k is too large."
            ps = d/self.k
            for i in xrange(self.k-1):
                selected = np.array([False] * d)
                selected[i * ps : (i+1) * ps] = True
                self.features.append(selected)
            selected = np.array([False] * d)
            selected[(i+1) * ps :] = True
            self.features.append(selected)
        else:
            pass
    
    def convolve(self, img, kw, ss):
        weights = np.ones(kw*kw).reshape(kw, kw)
        res = convolve(img, weights, mode='constant')/float(kw * kw)
        choose = np.tile([True] + [False] * (ss - 1), self.width/ss)
        assert len(choose) == self.width, "stepSize doesn't match image's width"
        return res[choose][:, choose].flatten()
                       
    def transform(self, X, kw, ss):
        n = X.shape[0]
        res = np.zeros((n, (self.width/ss)**2))
        tmpX = X.reshape(n, self.width, self.width)
        for i, img in enumerate(tmpX):
            res[i] = self.convolve(img, kw, ss)
        return res
        
    def fit(self, X, Y):
        n = X.shape[0]
        predictY = np.zeros(n)
        self.betas = []
        if self.scheme == "convolve":
            for k in self.kernels:
                #print "kernel size: ", k[0], " kernel step: ", k[1]
                trainX = self.transform(X, k[0], k[1])
                self.clf.fit(trainX, Y - predictY)
                self.betas.append(self.clf.getBeta())
                predictY += self.clf.predict(trainX)
        else:
            self.generateFeatures(X)
            for i in xrange(self.k):
                trainX = X[:, self.features[i]]
                self.clf.fit(trainX, Y - predictY)
                self.betas.append(self.clf.getBeta())
                predictY += self.clf.predict(trainX)
    
    def predict(self, X):
        n = X.shape[0]
        predictY = np.zeros(n)
        if self.scheme == "convolve":
            for i, k in enumerate(self.kernels):
                testX = self.transform(X, k[0], k[1])
                self.clf.setBeta(self.betas[i])
                predictY += self.clf.predict(testX)            
        else:
            for i in xrange(self.k):
                testX = X[:, self.features[i]]
                self.clf.setBeta(self.betas[i])
                predictY += self.clf.predict(testX)            
        return predictY
        
if __name__ == "__main__":
    xis = np.linspace(-1., 0, 11)
    regs = np.logspace(-3, 3, 7)
    mnist = fetch_mldata('MNIST original', data_home="data/")
    data = np.hstack((mnist.target.reshape(-1,1), mnist.data))
    trainX, trainY, testX, testY = Util.readData(data, intercept=False, testSize=0.65)
    gls = GLS()
    hda = HighDimAlg(gls, scheme="convolve")
    hda.setWidth(28)
    hda.setKernels([(2,2), (4,4), (7,7)])
    bestScore, bestXi, bestReg = 1e10, None, None        
    for xi in xis:
        hda.setXi(xi)
        print "Current xi = ", xi
        t0 = time.time()
        score, reg = Util.crossValidate(hda, trainX, trainY, \
                                        Util.brierScore, 5, "Regular", regs)
        t1 = time.time()
        print "CV interval: " + str(t1-t0) + "sec"
        if score < bestScore:
            bestScore, bestXi, bestReg = score, xi, reg
    print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
    hda.setXi(bestXi)
    hda.setRegular(bestReg)
    
    trainX, trainY, testX, testY = Util.readData(data, intercept=False)
    hda.fit(trainX, trainY)
    print Util.brierScore(hda.predict(testX), testY)