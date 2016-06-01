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
        self.miuXi = 0.
        
    def setSigmaBeta(self, sigmaBeta):
        self.sigmaBeta = sigmaBeta
        
    def setSigmaXi(self, sigmaXi):
        self.sigmaXi = sigmaXi

    def setMiuXi(self, miuXi):
        self.miuXi = miuXi        
        
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
               np.log(np.power(det, -0.5)) - 0.5 * beta.dot(inv).dot(beta) + \
               np.log(np.power(self.sigmaXi, -0.5)) - 1./(2*self.sigmaXi)*xi*xi
               
    def logpdf_miuXi(self, z):
        beta, xi = z[:-1], z[-1]
        det = np.prod(self.sigmaBeta)
        inv = np.diag(1./self.sigmaBeta)
        return self.loglikelihood(beta, xi) + \
               np.log(np.power(det, -0.5)) - 0.5 * beta.dot(inv).dot(beta) \
               - 0.5 * (xi - self.miuXi)**2
               
    def pdf(self, z):
        return np.exp(self.logpdf(z))
        
    def pdf_miuXi(self, z):
        return np.exp(self.logpdf_miuXi(z))

if __name__ == "__main__":
    trainX, trainY, testX, testY = Util.readData("data/harberman.data", testSize=0)
    
    D = trainX.shape[1]
    jd = JointDist(trainX, trainY)
    sigmaBeta = np.random.rand(D)
    sigmaXi = np.random.rand()
    miuXi = np.random.randn()
    print("miuXi = ", miuXi)
    Q = []
    #sigmaXis = []
    miuXis = []
    #beta = np.random.multivariate_normal([0.]*D, np.diag(sigmaBeta))
    #xi = np.array([np.sqrt(sigmaXi) * np.random.randn()])
    #init = np.concatenate((beta, xi))
    init = np.zeros(D + 1)
    
    #print "logpdf = ", jd.logpdf(init)
    #print "pdf = ", jd.pdf(init)
    
    iters = 1
    m = np.linspace(5000, 10000, iters)
    scala = 1e-4 #0.0001
    for i in range(iters):
        jd.setSigmaBeta(sigmaBeta)
        #jd.setSigmaXi(sigmaXi)
        jd.setMiuXi(miuXi)
        
        #samples = Sampling.Slice_Sampling_Multi(init, jd.pdf, 1000, 0.01, 10)
    #    qsigma = np.diag(scala * np.random.rand(D+1))
    #    samples, acceptRate = Sampling.MH_Sampling(init, jd.logpdf_miuXi, int(m[i]), int(m[i]*0.5), 1, qsigma)
        samples, acceptRate = Sampling.Gibbs_Sampling(init, jd.logpdf_miuXi, 50000, scala)
        #samples, acceptRate = Sampling.MTMI_Sampling(init, jd.pdf, 5, int(m[i]), int(m[i]*0.1), 20, qsigma)
        
        print("Accept rate: ", acceptRate)
        if acceptRate < 0.15:
            scala /= 2
            
        sigma = (samples**2).sum(axis=0)/float(samples.shape[0])
        sigmaBeta, sigmaXi = sigma[:-1], sigma[-1]
        miuXi = samples.mean(axis=0)[-1]
        
        q = np.mean([jd.logpdf(s) for s in samples])
        init = samples[-1]
        Q.append(q)
        #sigmaXis.append(sigmaXi)
        miuXis.append(miuXi)
    
     
    #plt.figure(1)
    #plt.subplot(211)
    #plt.plot(Q)
    #
    #plt.subplot(212)
    ##plt.plot(sigmaXis)
    #plt.plot(miuXis)
    #plt.show()       
       
    predY = np.zeros(testY.size).reshape(-1, 1)
    for s in samples:
        beta, xi = s[:-1], s[-1]
        v = testX.dot(beta.reshape(-1, 1))
        GEVFunc.clip(xi, v)
        predY += GEVFunc.inverseLink(xi, v)
    predY /= samples.shape[0]
    predY = predY.flatten()
    print("b = %f b1 = %f b0 = %f c = %f a = %f r = %f p = %f f = %f m = %f g = %f" % Util.evaluate(predY, testY))
    
    plt.plot(samples[:, -1].flatten())
    plt.show()
    print(samples.mean(axis=0)[-1])
    
