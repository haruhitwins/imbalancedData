# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:19:26 2015

@author: Haolin
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, miu, var):
    return np.power(2*np.pi*var, -0.5) * np.exp(-(x-miu)**2 / (2.*var))
    
def log_pdf(x):
    return np.log(1./3 * (gaussian(x, 1, 0.1) + gaussian(x, 2, 0.05) + gaussian(x, 3, 0.1)))

def pdf(x):
    return 1./3 * (gaussian(x, 1, 0.1) + gaussian(x, 2, 0.05) + gaussian(x, 3, 0.1))

def MH_Sampling(init, log_pdf, iters, burnin, Mstep, sigma):
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
    samples = samples[burnin:]
    index = np.array(range(samples.shape[0]))
    return samples[index % Mstep == 0]

def steppingOut(f, x, y, w, m):
    u, v = np.random.rand(), np.random.rand()
    L = x - w * u
    R = L + w
    j = math.floor(m * v)
    k = m - 1 - j
    while j > 0 and y < f(L):
        L -= w
        j -= 1
    while k > 0 and y < f(R):
        R += w
        k -= 1
    return (L, R)
    
def shrinkage(f, x, y, (L, R)):
    l, r = L, R
    while 1:
        u = np.random.rand()
        sample = l + u * (r - l)
        if y < f(sample):
            return sample
        if sample < x:
            l = sample
        else:
            r = sample
    
def Slice_Sampling(x, pdf, iters, w, m):
    samples = np.zeros(iters)
    for i in xrange(iters):
        y = np.random.rand() * pdf(x)
        (L, R) = steppingOut(pdf, x, y, w, m)
        s = shrinkage(pdf, x, y, (L, R))
        samples[i] = s
        x = s
    return samples

def steppingOut_multi(pdf, z, index, y, w, m):
    u, v = np.random.rand(), np.random.rand()
    x = z[index]
    L = x - w * u
    R = L + w
    j = math.floor(m * v)
    k = m - 1 - j
    tmp = z.copy()
    tmp[index] = L
    while j > 0 and y < pdf(tmp):
        L -= w
        tmp[index] = L
        j -= 1
    tmp[index] = R
    while k > 0 and y < pdf(tmp):
        R += w
        tmp[index] = R
        k -= 1
    return (L, R)

def shrinkage_multi(pdf, z, index, y, (L, R)):
    l, r = L, R
    tmp = z.copy()
#    print "y = ", y
#    tmp[index] = R
#    print "pdf(R) = ", pdf(tmp)
#    tmp[index] = L
#    print "pdf(L) = ", pdf(tmp)
    while 1:
        u = np.random.rand()
        sample = l + u * (r - l)
        #print sample, l, r, r-l
        tmp[index] = sample
        if y < pdf(tmp):
            return sample
        if sample < z[index]:
            l = sample
        else:
            r = sample
    
def Slice_Sampling_Multi(init, pdf, iters, w, m):
    D = init.size
    state = init
    samples = np.zeros((iters, D))
    for i in xrange(iters):
        #sample = np.zeros(D)
        for j in xrange(D):
            y = np.random.rand() * pdf(state)
            (L, R) = steppingOut_multi(pdf, state, j, y, w, m)
            s = shrinkage_multi(pdf, state, j, y, (L, R))
            #sample[j] = s
            state[j] = s
        #samples[i] = sample
        samples[i] = state
    return samples
   
if __name__ == "__main__":
#    init = np.array([np.random.rand()])
#    s = MH_Sampling(init, log_pdf, 5000, 1000, 10, 2 * np.eye(1))
    s = Slice_Sampling(np.random.rand(), pdf, 3000, 0.1, 10)
    n, bins, patches = plt.hist(s, 50, normed=1, facecolor='g', alpha=0.75)
    plt.axis([-1, 5, 0, 1])
    x = np.linspace(0, 4, 100)
    y = [pdf(a) for a in x]
    plt.plot(x,y,'r-')
    plt.show()
