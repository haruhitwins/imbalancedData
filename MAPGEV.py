# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 21:48:33 2016

@author: Haolin
"""

import numpy as np
import GEVFunc

class MAPGEV(object):
    def __init__(self, iterations=100, learningRate=0.03):
        self.iterations = iterations
        self.learningRate = learningRate

    def betaDeriva(self, X, Y, beta, xi):
        v = X.dot(beta).reshape(-1, 1)
        y = Y.reshape(-1, 1)
        GEVFunc.clip(xi, v)
        gev = GEVFunc.GEV(xi, v)
        loggev = GEVFunc.logGEV(xi, v)
        res = - np.sum(X * (loggev * (y - gev) / ((1 - gev) * (1 + xi*v))), axis=0) - beta
        return res
    
    def xiDeriva(self, X, Y, beta, xi):
        v = X.dot(beta).reshape(-1, 1)
        y = Y.reshape(-1, 1)
        GEVFunc.clip(xi, v)
        gev = GEVFunc.GEV(xi, v)
        loggev = GEVFunc.logGEV(xi, v)
        res = np.sum((np.log(1 + xi*v) / xi**2 - v / xi / (1 + xi*v)) 
               * loggev * (y - gev) / (1 - gev)) - xi
        return res
      
    def betaSecDeriva(self, X, Y, beta, xi):
        pos, neg = X[Y == 1], X[Y == 0]
        v = pos.dot(beta).reshape(-1, 1)
        GEVFunc.clip(xi, v)
        w = (-1-xi) * np.power(1 + xi * v, -1./xi-2)
        comp1 = pos.T.dot(np.diag(w.flatten())).dot(pos)
        v = neg.dot(beta).reshape(-1, 1)
        GEVFunc.clip(xi, v)
        gev = GEVFunc.GEV(xi, v)
        a = 1 + xi * v
        w = np.power(a, -1./xi-2)*gev/(1-gev)*(1+xi - np.power(a, -1./xi)/(1-gev))
        comp2 = neg.T.dot(np.diag(w.flatten())).dot(neg)
        return comp1 + comp2
    
    def fit(self, X, Y):
        n, d = X.shape
        beta = np.zeros(d)
        beta[-1] = np.log(-np.log(Y.mean()))
#        beta = np.array([ 2.20294651, -0.86604717, -0.10212129,  0.14875123])
        xi = 0.1
        xis = [xi]
        bs = [beta[-1]]
        for i in xrange(self.iterations):
            secdev = self.betaSecDeriva(X, Y, beta, xi)
            beta -= 0.1 * np.linalg.pinv(secdev).dot(self.betaDeriva(X, Y, beta, xi))
#            beta += self.learningRate/n * self.betaDeriva(X, Y, beta, xi)
            xi += self.learningRate * self.xiDeriva(X, Y, beta, xi)
            xis.append(xi)
            bs.append(beta[-1])
        return beta, xi, xis, bs
        
if __name__ == "__main__":
    estimator = MAPGEV()
    import Util
    trainX, trainY, testX, testY = Util.readData("data/german.data", testSize=0.3)

#    d = trainX.shape[1]
#    from BayesianAnalysis import JointDist
#    jd = JointDist(trainX, trainY)
#    jd.setSigmaBeta(np.ones(d))
#    jd.setMiuXi(0.)
#    beta = np.array([ 2.20294651, -0.86604717, -0.10212129,  0.14875123])
#    i = 8
#    s = []
#    while i < 10:
#        z = np.concatenate((beta, [i]))
#        s.append(jd.logpdf_miuXi(z))
#        i += 0.1
    beta, xi, xis, bs = estimator.fit(trainX, trainY)
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(211)
    plt.plot(bs)
    
    plt.subplot(212)
    plt.plot(xis)
    plt.show()       
    print beta, xi
    
    v = testX.dot(beta)
    GEVFunc.clip(xi, v)
    predY = GEVFunc.inverseLink(xi, v).flatten()
    print "b = %f b1 = %f b0 = %f c = %f a = %f r = %f p = %f f = %f m = %f g = %f" % Util.evaluate(predY, testY)