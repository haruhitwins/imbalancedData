# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:54:25 2016

@author: Haolin
"""

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    import Util
    trainX, trainY, testX, testY = Util.readData("data/harberman.data", intercept=False, testSize=0.3)
    family = sm.families.Binomial()
    glm = sm.GLM(trainY, sm.add_constant(trainX), family=family)
    res = glm.fit()
#    print "sm beta ", res.params
    clf = LogisticRegression()
    clf.fit(trainX, trainY)
#    print "sk beta ", clf.coef_.flatten(), clf.intercept_
    
    print("sm: ", Util.evaluate(res.predict(sm.add_constant(testX)), testY))
    print("sk: ", Util.evaluate(clf.predict_proba(testX)[:,1].flatten(), testY))