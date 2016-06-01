# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:58:37 2015

@author: Haolin
"""

import numpy as np
import GEVFunc

class GLS(object):
    def __init__(self, xi=-0.2567, reg=0., iterations=100, tolerance=1e-12):
        self.xi = xi
        self.reg = reg
        self.iterations = iterations
        self.tol = tolerance
        self._beta = None
        self.step = 1.0
        self.manualInitial = False

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
    
    def initialize(self, X, Y):
        d = X.shape[1]
        self._beta = np.zeros(d).reshape(-1, 1)
        self._beta[-1] = np.log(-np.log(Y.mean()))
        
    def calculateL(self, xi):
        if xi < -1: raise Exception("No L exists when xi < -1")
        return np.exp(-1.-xi)*np.power(1.+xi, 1.+xi)

    def firstDeriva(self, X, Y, beta):
        v = X.dot(beta)
        GEVFunc.clip(self.xi, v)
        Y_hat = GEVFunc.inverseLink(self.xi, v)
        return X.T.dot(Y_hat - Y) / X.shape[0] + self.reg * beta
        
    def oneStep(self, X, Y):
        n, d = X.shape[0], X.shape[1]
        Y = Y.reshape(-1, 1) 
        L = self.calculateL(self.xi)
        try:
            secOrd = np.linalg.pinv(X.T.dot(X) * L / n + self.reg * np.eye(d))
        except Exception as e:
            print(e.message)
        v = X.dot(self._beta)
        GEVFunc.clip(self.xi, v)
        Y_hat = GEVFunc.inverseLink(self.xi, v)
        firOrd = X.T.dot(Y_hat - Y) / n + self.reg * self._beta
        self._beta -= self.step * secOrd.dot(firOrd)
    
    def oneStep2(self, X, Y):
        Y = Y.reshape(-1, 1) 
        descentDir = - self.firstDeriva(X, Y, self._beta)
        al, au, k = 0., 1., 20
        while k > 0:
            a = (al+au)/2
            tmpBeta = self._beta + a * descentDir
            deriva = self.firstDeriva(X, Y, tmpBeta)
            if deriva.T.dot(descentDir) > 0:
                au = a
            else:
                al = a
            k -= 1
        self._beta += 1 * a * descentDir
    
    def oneStep3(self, X, Y):
        n, d = X.shape[0], X.shape[1]
        Y = Y.reshape(-1, 1) 
        v = X.dot(self._beta)
        GEVFunc.clip(self.xi, v)
        W = np.diag(GEVFunc.derivInverseLink(self.xi, v).flatten())
        secOrd = np.linalg.pinv(X.T.dot(W).dot(X) / n + self.reg * np.eye(d))
        Y_hat = GEVFunc.inverseLink(self.xi, v)
        firOrd = X.T.dot(Y_hat - Y) / n + self.reg * self._beta
        self._beta -= self.step * secOrd.dot(firOrd)    
    
    def fit(self, X, Y):
        n, d = X.shape[0], X.shape[1]
        Y = Y.reshape(-1, 1) 
        L = self.calculateL(self.xi)
        #secOrd = np.linalg.pinv((X.T.dot(X) * L + self.reg * np.eye(d)) / n)
#        I = np.eye(d)
#        I[-1, -1] = 0 #Not regularize the intercept
        try:
            secOrd = np.linalg.pinv(X.T.dot(X) * L / n + self.reg * np.eye(d))
        except Exception as e:
            print(e.message)
        if not self.manualInitial:
            self.initialize(X, Y)
        
        t = 0.
        while t < self.iterations:
            v = X.dot(self._beta)
            GEVFunc.clip(self.xi, v)
            Y_hat = GEVFunc.inverseLink(self.xi, v)
            #firOrd = (X.T.dot(Y_hat - Y) + self.reg * self._beta) / n  
#            tmp = self.reg * self._beta
#            tmp[-1] = 0
            firOrd = X.T.dot(Y_hat - Y) / n + self.reg * self._beta
            deltaBeta = self.step * secOrd.dot(firOrd)
            if t >= 100 and np.abs(deltaBeta).sum() < self.tol:
                self._beta -= deltaBeta
#                print "Converged. t = %d" % (t)
                break            
            self._beta -= deltaBeta
            t += 1
        self._beta = np.array(self._beta).flatten()
    
    def fit2(self, X, Y):
        Y = Y.reshape(-1, 1) 
        if not self.manualInitial:
            self.initialize(X, Y)
        t = 0.
        while t < self.iterations:
            descentDir = - self.firstDeriva(X, Y, self._beta)
            al, au, k = 0., 1., 10
            while k > 0:
                a = (al+au)/2
                tmpBeta = self._beta + a * descentDir
                deriva = self.firstDeriva(X, Y, tmpBeta)
                if deriva.T.dot(descentDir) > 0:
                    au = a
                else:
                    al = a
                k -= 1
            self._beta += self.step * a * descentDir
            t += 1
        self._beta = np.array(self._beta).flatten()
        
    def fit3(self, X, Y):
        n, d = X.shape[0], X.shape[1]
        Y = Y.reshape(-1, 1) 
        if not self.manualInitial:
            self.initialize(X, Y)
        t = 0.
        while t < self.iterations:
            v = X.dot(self._beta)
            GEVFunc.clip(self.xi, v)
            W = np.diag(GEVFunc.derivInverseLink(self.xi, v).flatten())
#            secOrd = np.linalg.pinv(X.T.dot(W).dot(X) / n + self.reg * np.eye(d))
            A = X.T.dot(W).dot(X) / n + self.reg * np.eye(d)
            Y_hat = GEVFunc.inverseLink(self.xi, v)
            b = X.T.dot(Y_hat - Y) / n + self.reg * self._beta
            self._beta += np.linalg.lstsq(A, -b)[0]
#            self._beta -= self.step * secOrd.dot(firOrd)    
            t += 1
        self._beta = np.array(self._beta).flatten()
    
    def predict(self, X):
        v = X.dot(self._beta)
        GEVFunc.clip(self.xi, v)
        return GEVFunc.inverseLink(self.xi, v)

if __name__ == "__main__": 
    import Util
    import matplotlib.pyplot as plt
    
    iters = 10
    dataCnt = 13
    fir = np.zeros(dataCnt*iters).reshape(dataCnt, iters)
    sec = np.zeros(dataCnt*iters).reshape(dataCnt, iters)
    myL = np.zeros(dataCnt*iters).reshape(dataCnt, iters)
    i = 0
#    for data in ['wisconsin','pima','glass0','haberman','vehicle0','segment0', \
#                 'ecoli-0-3-4_vs_5','yeast-0-3-5-9_vs_7-8','vowel0', \
#                 'led7digit-0-2-4-5-6-7-8-9_vs_1','page-blocks-1-3_vs_4', \
#                 'flare-F','car-good','winequality-red-8_vs_6','kr-vs-k-zero_vs_eight',\
#                 'poker-8_vs_6','abalone19']:
    '''
    haberman,ecoli-0-3-4_vs_5,vowel0,wisconsin 0.1 0.2
    glass0,segment0 -0.5 0.2
    '''
    f = plt.figure(num=1,figsize=(6,8))
    ax = f.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    f, axs = plt.subplots(3, 2, sharex=True, num=1)
    xaxis = range(iters)
    for data in ['ecoli-0-3-4_vs_5','haberman','glass0','vowel0','wisconsin','segment0']:
        print(data)
        trainX, trainY, testX, testY = Util.readData("data/processed/"+data+".dat",testSize=0.)
        
        if data in ['glass0', 'segment0']:
            xi = -0.5
        else:
            xi = 0.1
        reg = 0.2
        
        clf = GLS()
        clf.setRegular(reg)
        clf.setXi(xi)
        clf.initialize(trainX, trainY)
        bs1 = []
        for _ in range(iters):
            bs1.append(Util.logLoss(clf.predict(trainX), trainY))
            clf.oneStep2(trainX, trainY)
        fir[i] = bs1
        
        clf = GLS()
        clf.setRegular(reg)
        clf.setXi(xi)
        clf.initialize(trainX, trainY)
        bs2 = []
        for _ in range(iters):
            bs2.append(Util.logLoss(clf.predict(trainX), trainY))
            clf.oneStep3(trainX, trainY)
        sec[i] = bs2
        
        clf = GLS()
        clf.setRegular(reg)
        clf.setXi(xi)
        clf.initialize(trainX, trainY)
        bsL = []
        for _ in range(iters):
            bsL.append(Util.logLoss(clf.predict(trainX), trainY))
            clf.oneStep(trainX, trainY)
        myL[i] = bsL

        l1, l2, l3 = axs[i/2][i%2].plot(xaxis, bs1, xaxis, bs2, xaxis, bsL)
        axs[i/2][i%2].set_title(data)
        plt.setp(l1, c='r', ls='dotted', lw=3.0)
        plt.setp(l2, c='b', ls='--', lw=3.0)
        plt.setp(l3, c='g', ls='-', lw=2.0)
        if i == 1:
            axs[i/2][i%2].legend([l1,l2,l3],['gradient',"Hessian",'proposed'])#,'upper right')
        axs[i/2][i%2].grid(True)
        i += 1    
    ax.set_ylabel('Log loss', fontsize=14)
    ax.set_xlabel('Iterations', fontsize=14)
    plt.savefig("./convergence.pdf", bbox_inches='tight')
    plt.show()    
    
#    xaxis = range(iters)
#    l1, l2, l3 = plt.plot(xaxis, fir.mean(axis=0), xaxis, sec.mean(axis=0), xaxis, myL.mean(axis=0))
#    plt.setp(l1, c='r', ls='dotted', lw=3.0)
#    plt.setp(l2, c='g', ls='--', lw=3.0)
#    plt.setp(l3, c='b', ls='-.', lw=3.0)
#    plt.grid(True)
#    plt.ylabel('Brier score')
#    plt.xlabel('Iterations')
#    plt.show()
    