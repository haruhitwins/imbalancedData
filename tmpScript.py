# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 14:13:17 2015

@author: Haolin
"""

import numpy as np
import time
import Util
from GLS import GLS

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = range(10)
    y = range(10)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(x,y)
    ax2.plot(x,y)    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('common xlabel')
    ax.set_ylabel('common ylabel')
    plt.show()
#    iters = 100
#    xi = 0.1
#    reg = 0.2
#    ts = np.zeros(3*17).reshape(17,3)
#    i = 0
#    for data in ['wisconsin','pima','glass0','haberman','vehicle0','segment0', \
#                 'ecoli-0-3-4_vs_5','yeast-0-3-5-9_vs_7-8','vowel0', \
#                 'led7digit-0-2-4-5-6-7-8-9_vs_1','page-blocks-1-3_vs_4', \
#                 'flare-F','car-good','winequality-red-8_vs_6','kr-vs-k-zero_vs_eight',\
#                 'poker-8_vs_6','abalone19']:
#        print data
#        trainX, trainY, testX, testY = Util.readData("data/processed/"+data+".dat",testSize=0.3)
#        
#        t = time.time()
#        for _ in xrange(10):
#            clf = GLS()
#            clf.setRegular(reg)
#            clf.setXi(xi)
#            clf.fit2(trainX, trainY)
#        ts[i, 0] = (time.time() - t)/10.
#        
#        t = time.time()
#        for _ in xrange(10):
#            clf = GLS()
#            clf.setRegular(reg)
#            clf.setXi(xi)
#            clf.fit3(trainX, trainY)
#        ts[i, 1] = (time.time() - t)/10.
#        
#        t = time.time()
#        for _ in xrange(10):
#            clf = GLS()
#            clf.setRegular(reg)
#            clf.setXi(xi)
#            clf.fit(trainX, trainY)
#        ts[i, 2] = (time.time() - t)/10.
#        i += 1
    