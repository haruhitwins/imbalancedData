# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:19:26 2015

@author: Haolin
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, miu, var):
    return np.power(2*np.pi*var, -0.5) * np.exp(-(x-miu)**2 / (2.*var))
    
def log_pdf(x):
    return np.log(1./3 * (gaussian(x, 1, 0.1) + gaussian(x, 2, 0.05) + gaussian(x, 3, 0.1)))

def pdf(x):
    return 1./3 * (gaussian(x, 1, 0.1) + gaussian(x, 2, 0.05) + gaussian(x, 3, 0.1))

def MH_Sampling(init, log_pdf, iters, sigma):
    assert init.shape[0] == 1   
    D = init.size
    samples = np.zeros((iters, D))
    state = init
    lp = log_pdf(state)
    acceptance = 0
    for i in xrange(iters):
        prop = np.random.multivariate_normal(state, sigma)
        lpp = log_pdf(prop)
        if np.log(np.random.rand()) < lpp - lp:
            state = prop
            lp = lpp
            acceptance += 1
        samples[i, :] = state
    print "Accept rate: %f" % (acceptance/float(iters))
    return samples

init = np.array([np.random.rand()])
s = MH_Sampling(init, log_pdf, 5000, 2 * np.eye(1))
s = s[500:] #Ommit burn-in
n, bins, patches = plt.hist(s, 50, normed=1, facecolor='g', alpha=0.75)
plt.axis([-1, 5, 0, 1])
x = np.linspace(0, 4, 100)
y = [pdf(a) for a in x]
plt.plot(x,y,'r-')
plt.show()
#print 0.04*sum(y)