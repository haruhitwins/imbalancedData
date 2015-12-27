# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:59:29 2015

@author: Haolin
"""

import numpy as np
import GEVFunc, Util, Sampling
import matplotlib.pyplot as plt

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
               
    def pdf(self, z):
        return np.exp(self.logpdf(z))


trainX, trainY, testX, testY = Util.readData("data/pima.data")
D = trainX.shape[1]
jd = JointDist(trainX, trainY)
sigmaBeta = np.random.rand(D)
sigmaXi = np.random.rand()
Q = []

#beta = np.random.multivariate_normal([0.]*D, np.diag(sigmaBeta))
#xi = np.array([np.sqrt(sigmaXi) * np.random.randn()])
#init = np.concatenate((beta, xi))
init = np.zeros(D + 1)
#qsigma = 0.0005*np.eye(D+1)

print "logpdf = ", jd.logpdf(init)
print "pdf = ", jd.pdf(init)

for i in xrange(50):
    jd.setSigmaBeta(sigmaBeta)
    jd.setSigmaXi(sigmaXi)
    qsigma = np.diag(0.000005 * np.random.rand(D+1))
    samples = Sampling.MH_Sampling(init, jd.logpdf, 5000, 1000, 20, qsigma)
    #samples = Sampling.Slice_Sampling_Multi(init, jd.pdf, 1000, 0.01, 10)
    sigma = (samples**2).sum(axis=0)/float(samples.shape[0])
    sigmaBeta, sigmaXi = sigma[:-1], sigma[-1]    
    q = np.mean([jd.logpdf(s) for s in samples])
    #init = np.mean(samples, axis=0)
    #print "Q = ", q
    Q.append(q)
#    if (samples == samples[0]).all():
#        break
        
plt.plot(Q)
plt.show()    

predY = np.zeros(testY.size).reshape(-1, 1)
for s in samples:
    beta, xi = s[:-1], s[-1]
    v = testX.dot(beta.reshape(-1, 1))
    GEVFunc.clip(xi, v)
    predY += GEVFunc.inverseLink(xi, v)
predY /= samples.shape[0]
predY = predY.flatten()
print Util.evaluate(predY, testY)
print predY



