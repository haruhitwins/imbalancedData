# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 21:48:33 2016

@author: Haolin
"""

import numpy as np
import GEVFunc

class MAPGEV(object):
    def __init__(self, iterations=50, learningRate=0.1):
        self.iterations = iterations
        self.learningRate = learningRate
        self.beta = None
        self.xi = 0.

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
        return comp1 + comp2 - np.eye(beta.size)
    
    def xiSecDeriva(self, X, Y, beta, xi):
        v = X.dot(beta).reshape(-1, 1)
        GEVFunc.clip(xi, v)
        y = Y.reshape(-1, 1)
        a = 1 + xi*v
        gev = GEVFunc.GEV(xi, v)
        loggev = GEVFunc.logGEV(xi, v)
        y_gev = y - gev
        one_gev = 1 - gev
        y_gev_1_gev = y_gev/one_gev
        xia = xi * a
        xiv = xi * v
        lna = np.log(a)
        comp1 = (xiv * (3*a-1) - 2*lna*(a**2)) / (xi * xia**2) * y_gev_1_gev * loggev
        comp2 = (1/xi**2 * lna - v/xia) * ((y-1)*loggev*gev/one_gev**2 + y_gev_1_gev) * (v/xia-lna/xi**2)/np.power(a, 1./xi)
        return np.sum(comp1 + comp2) - 1
        
    
    def fit(self, X, Y):
        n, d = X.shape
        self.beta = np.zeros(d)
        self.beta[-1] = np.log(-np.log(Y.mean()))
        self.xi = 0.1
        xis = [self.xi]
        bs = [self.beta[-1]]
        for i in range(self.iterations):
            secdev = self.betaSecDeriva(X, Y, self.beta, self.xi)
            try:
                deltaBeta = np.linalg.pinv(secdev).dot(self.betaDeriva(X, Y, self.beta, self.xi))
            except Exception as e:
                print("%s. Numbers of iteraions: %d" % (e.message, i))
                break
            if np.isnan(deltaBeta).any():
                print("deltaBeta is nan. Numbers of iteraions: %d" % (i))
                break
            self.beta -= self.learningRate * deltaBeta
            deltaXi = self.xiDeriva(X, Y, self.beta, self.xi)/self.xiSecDeriva(X, Y, self.beta, self.xi)
            if np.isnan(deltaXi):
                print("deltaXi is nan. Numbers of iteraions: %d" % (i))
                break
            self.xi -= self.learningRate * deltaXi
#            beta += self.learningRate/n * self.betaDeriva(X, Y, beta, xi)
#            xi += self.learningRate * self.xiDeriva(X, Y, beta, xi)
            xis.append(self.xi)
            bs.append(self.beta[-1])
        return xis, bs
        
    def predict(self, X):
        v = X.dot(self.beta)
        GEVFunc.clip(self.xi, v)
        return GEVFunc.inverseLink(self.xi, v).flatten()
        
if __name__ == "__main__":
    estimator = MAPGEV()
    import Util
    trainX, trainY, testX, testY = Util.readData("data/processed/ecoli-0-1_vs_2-3-5.dat", testSize=0.3)

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
    xis, bs = estimator.fit(trainX, trainY)
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(211)
    plt.plot(bs)
    
    plt.subplot(212)
    plt.plot(xis)
    plt.show()       
    
    predY = estimator.predict(testX)
    print("b = %f b1 = %f b0 = %f c = %f a = %f r = %f p = %f f = %f m = %f g = %f" % Util.evaluate(predY, testY))