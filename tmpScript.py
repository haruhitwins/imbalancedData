# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 14:13:17 2015

@author: Haolin
"""

import numpy as np
#import time
import Util
from HDA import HighDimAlg
from GLS import GLS
from sklearn.linear_model import LogisticRegression
#from sklearn.datasets import fetch_mldata

def evaluate(predY, testY):
    b = Util.brierScore(predY, testY)
    c = Util.calibrationLoss(predY, testY)
    a = Util.auc(predY, testY)
    predY = Util.probToLabel(predY)
    r = Util.recall(predY, testY)
    p = Util.precision(predY, testY)
    f = Util.f1(predY, testY)
    return b,c,a,r,p,f

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

cifar = unpickle("data/cifar-10-batches-py/data_batch_1")
labels = np.array(cifar["labels"]).reshape(-1, 1)
data = np.hstack((labels, cifar["data"]))
for i in xrange(2,7,1):
    cifar = unpickle("data/cifar-10-batches-py/data_batch_"+str(i))
    labels = np.array(cifar["labels"]).reshape(-1, 1)
    tmp = np.hstack((labels, cifar["data"]))
    data = np.vstack((data, tmp))
trainX, trainY, testX, testY = Util.readData(data)

p, k = 1024, 3
bestScore, bestXi, bestReg = 1e10, None, None
xis = np.linspace(-1., 1.5, 26)
regs = np.logspace(-3, 3, 7)
clf = GLS() 
for xi in xis:       
    hda = HighDimAlg(clf, "random", p, k)
    hda.setXi(xi)
    print "Current xi = ", xi
    score, reg = Util.crossValidate(hda, trainX, trainY, \
                                    Util.brierScore, 4, "Regular", regs)
    if score < bestScore:
        bestScore, bestXi, bestReg = score, xi, reg
print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
hda.setXi(bestXi)
hda.setRegular(bestReg)
hdaTrain, hdaTest = [0]*6, [0]*6
for _ in xrange(10):
    print _
    trainX, trainY, testX, testY = Util.readData(data)
    hda.fit(trainX, trainY)
    predY = hda.predict(trainX)
    for i, v in enumerate(evaluate(predY, trainY)):
        hdaTrain[i] += v
    predY = hda.predict(testX)
    for i, v in enumerate(evaluate(predY, testY)):
        hdaTest[i] += v
hdaTrain = tuple([x/10. for x in hdaTrain])
hdaTest = tuple([x/10. for x in hdaTest])
with open("log/record.txt", 'a') as f:
    f.write("CIFAR-1 HDA preproc=False intercept=True patchSize=%d kTimes=%d xi=%.1f reg=%.1f\n" % (p, k, bestXi, bestReg))
    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % hdaTrain)
    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % hdaTest)

#clf = GLS()
#regs = np.logspace(-3, 3, 7)
#xis = np.linspace(-1., 1.5, 26)
#
#for name in ["german", "glass", "harberman", "pima", "vehicle", \
#             "trans_cleveland-0_vs_4", \
#             "trans_dermatology-6", \
#             "trans_lymphography-normal-fibrosis", \
#             "trans_page-blocks-1-3_vs_4", \
#             "trans_segment0", \
#             "trans_vowel0", \
#             "trans_zoo-3", "letter-A", \
#             "trans1_kddcup-buffer_overflow_vs_back", \
#             "trans1_kddcup-guess_passwd_vs_satan", \
#             "trans1_kddcup-land_vs_portsweep", \
#             "trans1_kddcup-land_vs_satan", \
#             "trans1_kddcup-rootkit-imap_vs_back", \
#             "trans2_kddcup-buffer_overflow_vs_back", \
#             "trans2_kddcup-guess_passwd_vs_satan", \
#             "trans2_kddcup-land_vs_portsweep", \
#             "trans2_kddcup-land_vs_satan", \
#             "trans2_kddcup-rootkit-imap_vs_back"]:
#    bestScore, bestXi, bestReg = 1e10, None, None
#    trainX, trainY, testX, testY = Util.readData("data/"+name+".data")
#    for xi in xis:
#        clf.setXi(xi)
#        score, reg = Util.crossValidate(clf, trainX, trainY, \
#                                        Util.brierScore, 5, "Regular", regs)
#        if score < bestScore:
#            bestScore, bestXi, bestReg = score, xi, reg
#    print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
#    clf.setXi(bestXi)
#    clf.setRegular(bestReg)
#    glsTrain, glsTest, lrTrain, lrTest = [0]*6, [0]*6, [0]*6, [0]*6
#    for _ in xrange(10):
#        trainX, trainY, testX, testY = Util.readData("data/"+name+".data")
#        clf.fit(trainX, trainY)
#        predY = clf.predict(trainX)
#        for i, v in enumerate(evaluate(predY, trainY)):
#            glsTrain[i] += v
#        predY = clf.predict(testX)
#        for i, v in enumerate(evaluate(predY, testY)):
#            glsTest[i] += v
#        
#        lg = LogisticRegression()
#        lg.fit(trainX, trainY)
#        predY = lg.predict_proba(trainX)[:,1].flatten()
#        for i, v in enumerate(evaluate(predY, trainY)):
#            lrTrain[i] += v
#        predY = lg.predict_proba(testX)[:,1].flatten()
#        for i, v in enumerate(evaluate(predY, testY)):
#            lrTest[i] += v
#    glsTrain = tuple([x/10. for x in glsTrain])
#    glsTest = tuple([x/10. for x in glsTest])
#    lrTrain = tuple([x/10. for x in lrTrain])
#    lrTest= tuple([x/10. for x in lrTest])
#    
#    with open("log/record.txt", 'a') as f:
#        f.write(name+" GLS preproc=False intercept=True\n")
#        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % glsTrain)
#        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % glsTest)
#        f.write("LogisticRegression\n")
#        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % lrTrain)
#        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n\n" % lrTest)
#        


#mnist = fetch_mldata('MNIST original', data_home="data/")
#chosenNum = 1
#mnist.target[mnist.target != chosenNum] = 0
#mnist.target[mnist.target == chosenNum] = 1
#data = np.hstack((mnist.target.reshape(-1,1), mnist.data))
#trainX, trainY, testX, testY = Util.readData(data)
#
#bestScore, bestXi, bestReg = 1e10, None, None
#xis = np.linspace(-1., -0.1, 10)
#regs = np.logspace(-3, 3, 7)
#
#for xi in xis:
#    clf = GLS()    
#    hda = HighDimAlg(clf, "random", 300, 3)
#    hda.setXi(xi)
#    print "Current xi = ", xi
#    score, reg = Util.crossValidate(hda, trainX, trainY, \
#                                    Util.brierScore, 5, "Regular", regs)
#    if score < bestScore:
#        bestScore, bestXi, bestReg = score, xi, reg
#print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
#clf = GLS()
#hda = HighDimAlg(clf, "random", 200, 6)
#hda.setXi(bestXi)
#hda.setRegular(bestReg)
#
#glsTrain, glsTest, lrTrain, lrTest = [0]*6, [0]*6, [0]*6, [0]*6
#for _ in xrange(10):
#    print _
#    trainX, trainY, testX, testY = Util.readData(data)
#    hda.fit(trainX, trainY)
#    predY = hda.predict(trainX)
#    for i, v in enumerate(evaluate(predY, trainY)):
#        glsTrain[i] += v
#    predY = hda.predict(testX)
#    for i, v in enumerate(evaluate(predY, testY)):
#        glsTest[i] += v
#    
#    lg = LogisticRegression()
#    lg.fit(trainX, trainY)
#    predY = lg.predict_proba(trainX)[:,1].flatten()
#    for i, v in enumerate(evaluate(predY, trainY)):
#        lrTrain[i] += v
#    predY = lg.predict_proba(testX)[:,1].flatten()
#    for i, v in enumerate(evaluate(predY, testY)):
#        lrTest[i] += v
#glsTrain = tuple([x/10. for x in glsTrain])
#glsTest = tuple([x/10. for x in glsTest])
#lrTrain = tuple([x/10. for x in lrTrain])
#lrTest= tuple([x/10. for x in lrTest])
#
#with open("log/record.txt", 'a') as f:
#    f.write("MNIST-1 HDA preproc=False intercept=True patchSize=200 kTimes=6\n")
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % glsTrain)
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % glsTest)
#    f.write("LogisticRegression\n")
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % lrTrain)
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n\n" % lrTest)
    
