# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:32:40 2015

@author: Haolin
"""

import numpy as np
import GEVFunc

class GEVCanReg(object):
    def __init__(self, xi=-0.2567, regular=1., iterations=50, tolerance=1e-10):
        self.xi = xi
        self.iterations = iterations
        self.tol = tolerance
        self.regular = regular
        self._eta = None
        self._beta = None
        self._v = None
        self._gamma = 1.

    def __str__(self):
        return "GEVCanReg xi = %f, iteraions = %d, tolerance= % f, regular = %f" \
                % (self.xi, self.iterations, self.tol, self.regular)

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
        return self._beta

    def getResults(self):
        return self._eta

    def __initialize(self, X, Y):
        self._eta = np.array([0.25] * Y.size)
        self._eta[Y == 1] = 0.75
        self._v = GEVFunc.link(self.xi, self._eta)

    def fit(self, X, Y):
        self.__initialize(X, Y)
        xi = self.xi
        t = 0
        while t < self.iterations:
            #Caculate weight matrix
            assert (self._eta > 0).all(), "eta value less than 0"
            w = self._eta*np.power(-np.log(self._eta), xi+1)
            assert not np.isnan(w).any(), "w isnan error."
            assert not np.isinf(w).any(), "w ifinf error."
            W = np.diag(w)

            #Z is used for updating beta
            tmp = GEVFunc.derivLink(xi, self._eta)
            assert not np.isnan(tmp).any(), "tmp isnan error."
            assert not np.isinf(tmp).any(), "tmp isinf error."

            Z = self._v + self._gamma \
                        *tmp \
                        *(Y - self._eta)
            assert not np.isnan(Z).any(), "Z isnan error."
            assert not np.isinf(Z).any(), "Z isinf error."

            #Update beta
            mat = np.matrix(X.T.dot(W).dot(X)) \
                    + np.eye(X.shape[1])*self.regular
            self._beta = mat.I.dot(X.T).dot(W).dot(Z).getA1()
            assert not np.isnan(self._beta).any(), "beta isnan error."
            assert not np.isinf(self._beta).any(), "beta isinf error."

            #Calculate v
            self._v = X.dot(self._beta)
            GEVFunc.clip(xi, self._v)
            assert not np.isnan(self._v).any(), "v isnan error."
            assert not np.isinf(self._v).any(), "v isinf error."

            #Judge if eta is convergent
            newEta = GEVFunc.inverseLink2(xi, self._v)
            assert not np.isnan(newEta).any(), "eta isnan error."
            assert not np.isinf(newEta).any(), "eta isinf error."
            if np.abs(newEta - self._eta).sum() < self.tol:
                self._eta = newEta
                break
            self._eta = newEta
            t += 1
        #print "Total iterations: ", t

    def predict(self, X):
        self._v = X.dot(self._beta)
        GEVFunc.clip(self.xi, self._v)
        return GEVFunc.inverseLink2(self.xi, self._v)

if __name__ == "__main__":
    reg = GEVCanReg()
    print reg
    xi = 0.5
    test = np.array([.1,.1,.1])
    v = reg.link(xi, test)
    print "data = ", test
    print "link(data) = ", v
    print "inverseLink(link(data)) = ", reg.inverseLink2(xi, v)

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