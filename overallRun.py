# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 16:33:32 2016

@author: Haolin
"""
import numpy as np
import time, os
import Util
#from sklearn import svm
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    log = []
    path = "F:/data/WinPython-64bit-2.7.9.5/SpyderWorkspace/imbalancedData/data/processed"
    for f in os.listdir(path):
        if f.find(".dat") != -1:
            print f
            scores = np.zeros(10 * 10).reshape(10, 10)
            t0 = time.time()
            for i in xrange(10):
                trainX, trainY, testX, testY = Util.readData(source=os.path.join(path, f), intercept=False)
                clf = LogisticRegression()
                clf.fit(trainX, trainY)
                scores[i] = Util.evaluate(clf.predict_proba(testX)[:,1].flatten(), testY)
            mean = ','.join([str(s) for s in scores.mean(axis=0)])
            std = ','.join([str(s) for s in scores.std(axis=0)])
            print "Time: %.5f" % (time.time() - t0)
            name = os.path.splitext(f)[0]
            log.append(name+','+mean+'\n')
            log.append(name+','+std+'\n')
    with open("log/logisticRegression.csv", 'a') as f:
        f.writelines(log)