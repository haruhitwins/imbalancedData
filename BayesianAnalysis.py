# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:59:29 2015

@author: Haolin
"""

import numpy as np
import GEVFunc

class JointDist(object):
    def __init__(self, X, targets):
        self.X = X
        self.targets = targets
        self.sigmaBeta = np.ones(X.shape[1])
        self.sigmaXi = 1.
        
    def setSigmaBeta(self, sigmaBeta):
        self.sigmaBeta = sigmaBeta
        
    def setSigmaXi(self, sigmaXi):
        self.sigmaXi = sigmaXi
        
    def loglikelihood(self, beta, xi):
        v = self.X.dot(beta.reshape(-1, 1))
        GEVFunc.clip(xi, v)
        res = GEVFunc.inverseLink(xi, v)
        res[self.targets == 0] = 1 - res[self.targets == 0]
        return sum([np.log(x) for x in res])
        
    def logpdf(self, z):
        beta, xi = z[:-1], z[-1]
        det = np.prod(self.sigmaBeta)
        inv = np.diag(1./self.sigmaBeta)
        return self.loglikelihood(beta, xi) + \
               np.log(np.power(det, -0.5)) - 0.5 * beta.dot(inv).dot(beta).reshape(-1, 1) + \
               np.log(np.power(self.sigmaXi, -0.5)) - 1./(2*self.sigmaXi)*xi*xi


X = np.array([1,0,1,1]).reshape(2,2)        
targets = np.array([0, 1])
beta = np.array([1,1])
xi = 1.
z = np.ones(3)
jd = JointDist(X, targets)
print jd.logpdf(z)