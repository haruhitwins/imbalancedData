# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:41:01 2015

@author: Haolin
"""

import threading, Util, time
from GLS import GLS
from HDA import HighDimAlg
import numpy as np
from sklearn.datasets import fetch_mldata

def worker(chosenNum):
    print("Current chosen number: ", chosenNum)
#    global validateX, validateY, trainX, trainY, regs, xis, lock, clfs
    clf = GLS()
    hda = HighDimAlg(clf, "convolve")       
    hda.setWidth(28)
    hda.setKernels([(2,2), (4,4), (7,7)])
    
    target = validateY.copy()
    target[target == chosenNum] = 999
    target[target != 999] = 0
    target[target == 999] = 1
    
    bestScore, bestXi, bestReg = 1e10, None, None 
    
    for xi in xis:
        hda.setXi(xi)
        score, reg = Util.crossValidate(hda, validateX, target, \
                                        Util.brierScore, 2, "Regular", regs)
        if score < bestScore:
            bestScore, bestXi, bestReg = score, xi, reg
    print("bestScore, bestXi, bestReg = ", bestScore, bestXi, bestReg)
    hda.setXi(bestXi)
    hda.setRegular(bestReg)
    
    target = trainY.copy()
    target[target == chosenNum] = 999
    target[target != 999] = 0
    target[target == 999] = 1
    hda.fit(trainX, target)
    
    res = {}
    res['num'] = chosenNum
    res['clf'] = hda
    res['validateScore'] = bestScore
    lock.acquire()
    try:
        clfs.append(res)
    finally:
        lock.release()
    

xis = np.linspace(-1., 0, 11)
regs = np.logspace(-3, 3, 7)
clfs = []
mnist = fetch_mldata('MNIST original', data_home="data/")
data = np.hstack((mnist.target.reshape(-1,1), mnist.data))
n = data.shape[0]
np.random.shuffle(data)
train, test = data[: int(n*0.7)], data[int(n*0.7):]
validate = train[: int(len(train)*0.5)]
trainX, trainY, validateX, validateY = train[:, 1:], train[:, 0], validate[:, 1:], validate[:, 0]

lock = threading.Lock()
threads = []

print("begin threading")
t0 = time.time()
for num in range(10):
    t = threading.Thread(target=worker, args=(num,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
t1 = time.time()
print("Total time: " + str(t1-t0) + "sec")

import cPickle as pickle
with open("log/hda_scaled01_mnist_list.txt", "wb") as f:
    pickle.dump(clfs, f)
#with open("log/hda_mnist_list.txt", "rb") as f:
#    clfs = pickle.load(f)
print(clfs)
forTest = []
for clf in clfs:
    forTest.append((clf['num'], clf['clf']))
forTest = [x[1] for x in sorted(forTest, key=lambda x: x[0])]
testX, testY = test[:, 1:], test[:, 0]
predY = Util.oneVsAll(testX, forTest)
print("accuracy = ", Util.accuracy(predY, testY))