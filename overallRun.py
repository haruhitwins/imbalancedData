# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 16:33:32 2016

@author: Haolin
"""
import numpy as np
import time, os
import Util, MAPGEV, GLS, SMOTE
#from sklearn import svm
#from sklearn.linear_model import LogisticRegression
#import statsmodels.api as sm
#from statsmodels.tools.sm_exceptions import PerfectSeparationError
if __name__ == "__main__":
    log = []
    path = "F:/data/WinPython-64bit-2.7.9.4/SpyderWorkspace/imbalancedData/data/processed"
    regs = np.logspace(-3, 3, 7)
    t0 = time.time()
    for f in os.listdir(path):
        if f.find(".dat") != -1:
#            print f
            if f != 'abalone19.dat': continue
            scores = np.zeros(10 * 10).reshape(10, 10)            
            for i in range(10):
                trainX, trainY, testX, testY = Util.readData(source=os.path.join(path, f))
                est = MAPGEV.MAPGEV(iterations=20)
                est.fit(trainX, trainY)
                beta, xi = est.beta.reshape(-1, 1), est.xi
                clf = GLS.GLS()
#                clf.manualInitial = True
                if xi < -1:
                    clf.setXi(-1)
                elif xi > 50:
                    clf.setXi(50)
                else:
                    clf.setXi(xi)
                print("xi = %f clf.xi = %f" % (xi, clf.xi))
                bestScore, bestReg = Util.crossValidate(clf, trainX, trainY, Util.brierScore, 5, "Regular", regs)
#                bestScore, bestReg = 1., 0.
#                for reg in regs:
#                    clf.setRegular(reg)
#                    clf.setBeta(beta)
#                    clf.fit(trainX, trainY)
#                    score = Util.brierScore(clf.predict(testX), testY)
#                    if score < bestScore:
#                        bestScore = score
#                        bestReg = reg
                print("bestScore = %f bestReg = %f" % (bestScore, bestReg))
                clf.setRegular(bestReg)
                clf.setBeta(beta)
                clf.manualInitial = True
                clf.fit(trainX, trainY)
                scores[i] = Util.evaluate(clf.predict(testX), testY)
            print(scores.mean(axis=0)[4])
#            for i in xrange(10):
#                trainX, trainY, testX, testY = Util.readData(source=os.path.join(path, f))
#                '''
#                SMOTE begin
#                '''
#                train = SMOTE.Smote(k=4).fit(np.hstack((trainY.reshape(-1, 1), trainX)))
#                trainX, trainY = train[:, 1:], train[:, 0]
#                '''
#                SMOTE end
#                '''
#                link = sm.genmod.families.links.probit
#                family = sm.families.Binomial(link)
#                glm = sm.GLM(trainY, trainX, family=family)
#                try:
#                    clf = glm.fit()
#                    scores[i] = Util.evaluate(clf.predict(testX), testY)
#                except PerfectSeparationError, e:
#                    print '%d: %s' % (i, e.message)
#            mean = ','.join([str(s) for s in scores.mean(axis=0)])
#            std = ','.join([str(s) for s in scores.std(axis=0)])            
#            name = os.path.splitext(f)[0]
#            if (scores.mean(axis=0) == 0).all() :
#                log.append(name+' Perfect separation error.\n')
#                log.append(name+' Perfect separation error.\n')
#            else:
#                log.append(name+','+mean+'\n')
#                log.append(name+','+std+'\n')

#    print "Time: %.5f" % (time.time() - t0)
#    with open("log/SMOTE_GLM_probitRegression.csv", 'a') as f:
#        f.writelines(log)