# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 14:13:17 2015

@author: Haolin
"""

import numpy as np
import time
import Util
from HDA import HighDimAlg
from GLS import GLS

from sklearn.datasets import fetch_mldata
#from sklearn.datasets import fetch_20newsgroups_vectorized

#def unpickle(file):
#    import cPickle
#    fo = open(file, 'rb')
#    dict = cPickle.load(fo)
#    fo.close()
#    return dict
#
#cifar = unpickle("data/cifar-10-batches-py/data_batch_1")
#labels = np.array(cifar["labels"]).reshape(-1, 1)
#data = np.hstack((labels, cifar["data"]))
#for i in xrange(2,7,1):
#    cifar = unpickle("data/cifar-10-batches-py/data_batch_"+str(i))
#    labels = np.array(cifar["labels"]).reshape(-1, 1)
#    tmp = np.hstack((labels, cifar["data"]))
#    data = np.vstack((data, tmp))
#trainX, trainY, testX, testY = Util.readData(data)   
#
#print "Read file done. Fitting..."
#lr = LogisticRegression()
#lr.fit(trainX, trainY)
#print "Fit done. Evaluating..."
#lrTrain, lrTest = [0]*6, [0]*6
#predY = lr.predict_proba(trainX)[:,1].flatten()
#for i, v in enumerate(evaluate(predY, trainY)):
#    lrTrain[i] += v
#predY = lr.predict_proba(testX)[:,1].flatten()
#for i, v in enumerate(evaluate(predY, testY)):
#    lrTest[i] += v
#lrTrain = tuple([x/10. for x in lrTrain])
#lrTest= tuple([x/10. for x in lrTest])
#with open("log/record.txt", 'a') as f:
#    f.write("LogisticRegression\n")
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % lrTrain)
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n\n" % lrTest)

#train = fetch_20newsgroups_vectorized(subset='train', remove=('headers', 'footers', 'quotes'), data_home="data/")
#trainX = train.data
#trainY = train.target
#trainY[trainY != 1] = 0
#test = fetch_20newsgroups_vectorized(subset='test', remove=('headers', 'footers', 'quotes'), data_home="data/")
#testX = test.data
#testY = test.target
#testY[testY != 1] = 0
#p, k = 1024, 100
#bestScore, bestXi, bestReg = 1e10, None, None
#xis = np.linspace(-1., 1.5, 26)
#regs = np.logspace(-3, 3, 7)
#clf = GLS() 
#for xi in xis:       
#    hda = HighDimAlg(clf, "random", p, k)
#    hda.setXi(xi)
#    print "Current xi = ", xi
#    score, reg = Util.crossValidate(hda, trainX, trainY, \
#                                    Util.brierScore, 4, "Regular", regs)
#    if score < bestScore:
#        bestScore, bestXi, bestReg = score, xi, reg
#print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
#hda.setXi(bestXi)
#hda.setRegular(bestReg)
#hdaTrain, hdaTest = [0]*6, [0]*6
#for _ in xrange(10):
#    print _
#    #trainX, trainY, testX, testY = Util.readData(data)
#    hda.fit(trainX, trainY)
#    predY = hda.predict(trainX)
#    for i, v in enumerate(evaluate(predY, trainY)):
#        hdaTrain[i] += v
#    predY = hda.predict(testX)
#    for i, v in enumerate(evaluate(predY, testY)):
#        hdaTest[i] += v
#hdaTrain = tuple([x/10. for x in hdaTrain])
#hdaTest = tuple([x/10. for x in hdaTest])
#with open("log/record.txt", 'a') as f:
#    f.write("20news HDA preproc=False intercept=True patchSize=1024 kTimes=100\n")
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % hdaTrain)
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % hdaTest)

clf = GLS()
regs = np.logspace(-4, 4, 10)
xis = np.linspace(-1., 2., 31)
for name in ["german", "glass", "harberman", "pima", "vehicle", \
             "trans_cleveland-0_vs_4", \
             "trans_dermatology-6", \
             "trans_lymphography-normal-fibrosis", \
             "trans_page-blocks-1-3_vs_4", \
             "trans_segment0", \
             "trans_vowel0", \
             "trans_zoo-3", "letter-A", \
             "trans1_kddcup-buffer_overflow_vs_back", \
             "trans1_kddcup-guess_passwd_vs_satan", \
             "trans1_kddcup-land_vs_portsweep", \
             "trans1_kddcup-land_vs_satan", \
             "trans1_kddcup-rootkit-imap_vs_back", \
             "trans2_kddcup-buffer_overflow_vs_back", \
             "trans2_kddcup-guess_passwd_vs_satan", \
             "trans2_kddcup-land_vs_portsweep", \
             "trans2_kddcup-land_vs_satan", \
             "trans2_kddcup-rootkit-imap_vs_back"]:

    trainX, trainY, testX, testY = Util.readData("data/"+name+".data")
#    bestScore, bestXi, bestReg = 1e10, None, None
#    for xi in xis:
#        clf.setXi(xi)
#        score, reg = Util.crossValidate(clf, trainX, trainY, \
#                                        Util.brierScore, 5, "Regular", regs)
#        if score < bestScore:
#            bestScore, bestXi, bestReg = score, xi, reg
#    print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
#    clf.setXi(bestXi)
#    clf.setRegular(bestReg)
#    glsTrain, glsTest = [0]*6, [0]*6
    lrTrain, lrTest = [0]*6, [0]*6
    #from sklearn.linear_model import LogisticRegressionCV
    from sklearn import svm
    lr = svm.SVC(kernel="linear", probability=True)
    lr.fit(trainX, trainY)
    for _ in xrange(10):
        trainX, trainY, testX, testY = Util.readData("data/"+name+".data")
#        clf.fit(trainX, trainY)
#        predY = clf.predict(trainX)
#        for i, v in enumerate(Util.evaluate(predY, trainY)):
#            glsTrain[i] += v
#        predY = clf.predict(testX)
#        for i, v in enumerate(Util.evaluate(predY, testY)):
#            glsTest[i] += v
                
        predY = lr.predict_proba(trainX)[:,1].flatten()
        for i, v in enumerate(Util.evaluate(predY, trainY)):
            lrTrain[i] += v
        predY = lr.predict_proba(testX)[:,1].flatten()
        for i, v in enumerate(Util.evaluate(predY, testY)):
            lrTest[i] += v
#    glsTrain = tuple([x/10. for x in glsTrain])
#    glsTest = tuple([x/10. for x in glsTest])
    lrTrain = tuple([x/10. for x in lrTrain])
    lrTest= tuple([x/10. for x in lrTest])
    
    with open("log/tmp.txt", 'a') as f:
#        f.write(name+" GLS preproc=False intercept=True\n")
#        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % glsTrain)
#        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n\n" % glsTest)
        f.write(name+" SVM linear\n")
        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % lrTrain)
        f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n\n" % lrTest)
        

#xis = np.linspace(-1., 0, 11)
#regs = np.logspace(-3, 3, 7)
#clfs = []
#mnist = fetch_mldata('MNIST original', data_home="data/")
#data = np.hstack((mnist.target.reshape(-1,1), mnist.data))
#trainX, trainY, testX, testY = Util.readData(data,testSize=0.5)
#bestScore, bestXi, bestReg = 1e10, None, None        
#for xi in xis:
#    clf = GLS()    
#    hda = HighDimAlg(clf, "random", 512, 3)
#    hda.setXi(xi)
#    print "Current xi = ", xi
#    score, reg = Util.crossValidate(hda, trainX, trainY, \
#                                    Util.brierScore, 5, "Regular", regs)
#    if score < bestScore:
#        bestScore, bestXi, bestReg = score, xi, reg
#print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg
#clf = GLS()
#hda = HighDimAlg(clf, "random", 300, 3)
#hda.setXi(bestXi)
#hda.setRegular(bestReg)

#n = data.shape[0]
#data = np.hstack((data, np.ones(n).reshape(-1, 1)))
#np.random.shuffle(data)
#train, test = data[: int(n*0.7)], data[int(n*0.7):]
#validate = train[: int(len(train)*0.5)]
#trainX, trainY = train[:, 1:], train[:, 0]
#validateX, validateY = validate[:, 1:], validate[:, 0]
#clf = GLS()
#hda = HighDimAlg(clf, "convolve")       
#hda.setWidth(28)
#hda.setKernels([(2,2), (4,4), (7,7)])
#for chosenNum in xrange(10):
#    print "Current chosenNum = ", chosenNum
#    target = validateY.copy()
#    target[target == chosenNum] = 999
#    target[target != 999] = 0
#    target[target == 999] = 1
#    
#    bestScore, bestXi, bestReg = 1e10, None, None 
#    
#    for xi in xis:
#        hda.setXi(xi)
#        print "Current xi = ", xi
#        t0 = time.time()
#        score, reg = Util.crossValidate(hda, validateX, target, \
#                                        Util.brierScore, 5, "Regular", regs)
#        t1 = time.time()
#        print "CV interval: " + str(t1-t0) + "sec"
#        if score < bestScore:
#            bestScore, bestXi, bestReg = score, xi, reg
#    print "bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg    
#    hda.setXi(bestXi)
#    hda.setRegular(bestReg)
#    
#    target = trainY.copy()
#    target[target == chosenNum] = 999
#    target[target != 999] = 0
#    target[target == 999] = 1
#    hda.fit(trainX, target)
#    clfs.append(hda)
#
#import cPickle as pickle
#with open("log/hda_mnist_list.txt", "wb") as f:
#    pickle.dump(clfs, f)
#with open("log/hda_mnist_list.txt", "rb") as f:
#    clfs = pickle.load(f)

#testX, testY = test[:, 1:], test[:, 0]
#predY = Util.oneVsAll(testX, clfs)
#print "accuracy = ", Util.accuracy(predY, testY)
 
#glsTrain, glsTest = [0]*6, [0]*6
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
#glsTrain = tuple([x/10. for x in glsTrain])
#glsTest = tuple([x/10. for x in glsTest])
#
#with open("log/record.txt", 'a') as f:
#    f.write("MNIST-1 HDA preproc=False intercept=True patchSize=512 kTimes=3\n")
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n" % glsTrain)
#    f.write("b=%.8f, c=%.8f, a=%.8f, r=%.8f, p=%.8f, f=%.8f\n\n" % glsTest)

