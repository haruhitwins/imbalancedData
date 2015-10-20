# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:32:40 2015

@author: Haolin
"""

import numpy as np
from Util import assure

class GEVCanReg(object):
    def __init__(self, xi = -0.2567, iterations = 30, tolerance = 1e-8, regular = 1.):
        self.xi = xi
        self.iterations = iterations
        self.tol = tolerance
        self.regular = regular
        self.eta = None
        self.beta = None
        self.v = None
        self.gamma = 1.
    
    def __str__(self):
        return "GEVCanReg xi = %f iteraions = %d" \
                % (self.xi, self.iterations)
        
    def __repr__(self):
        return "GEV-Canonical Regression"
    
    def getXi(self):
        return self.xi
        
    def setXi(self, xi):
        self.xi = xi
    
    def getIterations(self):
        return self.iterations
    
    def setIterations(self, iterations):
        self.iterations = iterations
        
    def getTolerance(self):
        return self.tol
        
    def setTolerance(self, tol):
        self.tol = tol
    
    def getRegular(self):
        return self.regular
        
    def setRegular(self, regular):
        self.regular = regular
        
    def getBeta(self):
        return self.beta
        
    def getResults(self):
        return self.eta
        
    def link(self, xi, eta):
        assure((eta > 0).all(), "link input error.")
        if xi == 0:
            return -np.log(-np.log(eta))
        return (1./np.power(-np.log(eta), xi) - 1) / xi
    
    def inverseLink(self, xi, v):
        assure((1+v*xi >= 0).all(), "inverseLink input error.")
        return np.exp(-np.power((1+v*xi), -1./xi))
    
    def derivLink(self, xi, eta):
        assure((eta > 0).all(), "derivLink input error.")
        return 1./(eta*np.power(-np.log(eta), xi+1))
 
    def clip(self, xi, v):
        if xi > 0:
            v[v < -1./xi] = -1./xi
        elif xi < 0:
            v[v > -1./xi] = -1./xi
    
    def __initialize(self, X, Y):
        self.eta = np.array([0.25] * Y.size)
        self.eta[Y == 1] = 0.75
        self.v = self.link(self.xi, self.eta)
        
    def fit(self, X, Y):
        self.__initialize(X, Y)
        t = 0
        tmpY = Y.copy()
        tmpY[Y != 1] = 0
        while t < self.iterations:
            #Caculate weight matrix
            assure((self.eta > 0).all(), "eta value less than 0")
            w = self.eta*np.power(-np.log(self.eta), self.xi+1)
            assure(not np.isnan(w).any(), "w error.")
            #w[w < 1e-15] = 1e-15
            W = np.diag(w)
            
            #Z is used for updating beta
            tmp = self.derivLink(self.xi, self.eta)
            assure(not np.isnan(self.v).any(), "v error.")
            np.savetxt("v.txt", self.v)
            assure(not np.isnan(tmp).any(), "tmp error.")
            np.savetxt("tmp.txt", tmp)
            assure(not np.isnan(self.eta).any(), "eta error.")
            np.savetxt("eta.txt", self.eta)
            Z = self.v + self.gamma \
                        *tmp \
                        *(tmpY - self.eta)
            assure(not np.isnan(Z).any(), "Z error.")
            
            #Update beta
            mat = np.matrix(X.T.dot(W).dot(X)) \
                    + np.eye(X.shape[1])*self.regular
            self.beta = mat.I.dot(X.T).dot(W).dot(Z).getA1()
            assure(not np.isnan(self.beta).any(), "beta error.")
            #Calculate v
            self.v = X.dot(self.beta)
            self.clip(self.xi, self.v)
            
            #Judge if eta is convergent
            newEta = self.inverseLink(self.xi, self.v)
            #newEta[newEta < 1e-15] = 1e-15
            if np.abs(newEta - self.eta).sum() < self.tol:
                self.eta = newEta
                break
            self.eta = newEta
            t += 1
        print "Total iterations: ", t

    def predict(self, X):
        self.v = X.dot(self.beta)
        self.clip(self.xi, self.v)
        return self.inverseLink(self.xi, self.v)
    
if __name__ == "__main__":
    reg = GEVCanReg()
    print reg
    xi = 0.5
    test = np.array([.1,.1,.1])
    v = reg.link(xi, test)
    print "data = ", test
    print "link(data) = ", v
    print "inverseLink(link(data)) = ", reg.inverseLink(xi, v)
    
    X = np.array([[1,2,3,3],
                  [4,2,4,4],
                  [3,3,3,5],
                  [6,9,7,6],
                  [7,8,9,7],
                  [5,7,6,8]])
    Y = np.array([1,1,1,-1,-1,-1])
    reg.fit(X,Y)
    print "eta: ", reg.getResults()
    print "predict: ", reg.predict(X)