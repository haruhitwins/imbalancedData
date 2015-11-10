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
#from sklearn.datasets import fetch_mldata
#from sklearn.linear_model import LogisticRegression

#mnist = fetch_mldata('MNIST original', data_home="data/")
#
#chosenNum = 1
#mnist.target[mnist.target == chosenNum] = 1
#mnist.target[mnist.target != chosenNum] = 0
#data = np.hstack((mnist.target.reshape(-1,1), mnist.data))
for name in ["vehicle", "german", "glass", "harberman", "pima", "letter-A"]:
    trainX, trainY, testX, testY = Util.readData("data/"+name+".data", intercept=False)
    ratio = np.linspace(0.1, 1, 10)
    xis = np.linspace(-1., 1.5, 26)
    clf = GLS(reg=10.)
    irL = []
    xiL = []               
    train = np.hstack((trainY.reshape(-1, 1), trainX))
    pos = train[trainY == 1, :]
    neg = train[trainY != 1, :]
    pn = pos.shape[0]
    nn = neg.shape[0]
    
    for r in ratio:
        tn = int(pn * r)
        select = np.array([True] * tn + [False] * (pn - tn))
        np.random.shuffle(select)
        train = np.vstack((pos[select, :], neg))
        trainX, trainY = train[:, 1:], train[:, 0]
        print "IR = ", nn/float(tn)
        irL.append(nn/float(tn))    
        score, xi = Util.crossValidate(clf, trainX, trainY, \
                                       Util.brierScore, 5, "Xi", xis)
        print "bestXi = ", xi
        xiL.append(xi)
        clf.setXi(xi)
        clf.fit(trainX, trainY)
        predY = clf.predict(testX)
        print "brierScore = ", Util.brierScore(predY, testY)
        print "calibrationLoss = ", Util.calibrationLoss(predY, testY)
        print "auc = ", Util.auc(predY, testY)
        predY = Util.probToLabel(predY)
        print "recall = ", Util.recall(predY, testY)
        print "precision = ", Util.precision(predY, testY)
        print "f1 = ", Util.f1(predY, testY)
    with open("log/record.txt", 'a') as f:
        f.write(name + " reg = 10.0 preproc = False intercept = False\n")
        f.write("IR: " + str(irL) + '\n')
        f.write("XI: " + str(xiL) + '\n')
        