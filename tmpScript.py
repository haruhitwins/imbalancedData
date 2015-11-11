# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 14:13:17 2015

@author: Haolin
"""

import numpy as np
#import time
import Util
#from HDA import HighDimAlg
from GLS import GLS
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression

mnist = fetch_mldata('MNIST original', data_home="data/")

chosenNum = 1
mnist.target[mnist.target == chosenNum] = 1
mnist.target[mnist.target != chosenNum] = 0
data = np.hstack((mnist.target.reshape(-1,1), mnist.data))
trainX, trainY, testX, testY = Util.readData(data, intercept=False)
train = np.hstack((trainY.reshape(-1, 1), trainX))
pos = train[trainY == 1, :]
neg = train[trainY != 1, :]
pn = pos.shape[0]
np.random.shuffle(neg)
neg = neg[:pn]
nn = neg.shape[0]

ratio = np.linspace(200, 10, 10)
xis = np.linspace(-1., 1.5, 26)
clf = GLS(reg=0.)
irL = []
#xiL = []              
scL = []
for ir in ratio:
    tn = int(nn / ir)
    select = np.array([True] * tn + [False] * (pn - tn))
    np.random.shuffle(select)
    train = np.vstack((pos[select, :], neg))
    trainX, trainY = train[:, 1:], train[:, 0]
    print "IR = ", nn/float(tn)
    irL.append(nn/float(tn))    
#    score, xi = Util.crossValidate(clf, trainX, trainY, \
#                                   Util.brierScore, 5, "Xi", xis)
#    print "bestXi = ", xi
#    xiL.append(xi)
#    clf.setXi(xi)
#    clf.fit(trainX, trainY)
    clf = LogisticRegression()
    clf.fit(trainX, trainY)
    predY = clf.predict_proba(testX)[:,1].flatten()
    b = Util.brierScore(predY, testY)
    c = Util.calibrationLoss(predY, testY)
    a = Util.auc(predY, testY)
    predY = Util.probToLabel(predY)
    r = Util.recall(predY, testY)
    p = Util.precision(predY, testY)
    f = Util.f1(predY, testY)
    scores = "b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f" % (b,c,a,r,p,f)
    print scores
    scL.append(scores+'\n')
with open("log/record.txt", 'a') as f:
    f.write("MNIST(1:1) LogisticRegression\n")
    f.write("IR: " + str(irL) + '\n')
    #f.write("XI: " + str(xiL) + '\n')
    f.writelines(scL)