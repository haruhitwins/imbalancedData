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

def helperGaussian(x, sigma):
    D = x.size
    diag = sigma.diagonal()
    return np.exp(-sum(x * (1.0/diag) * x)/2.0) * np.power(2*np.pi, -D/2.0) * (1.0 / np.sqrt(np.prod(diag)))

def proportionProb(w):
    """
    Draw an index from range 0 to len(w)-1 according to probablities
    proportianl to components in vector w.
    """
    k = w.size
    n = np.random.rand() * sum(w)
    i, summation = 0, 0
    while i < k:
        summation += w[i]
        if n <= summation:
            return i
        i += 1
    print("Never reach here")
    return k

def MTMI_Sampling(init, pdf, k, iters, burnin, Mstep, sigma):
    """
    Multiple-trial Metropolized independent sampler.
    This sampler doesn't work well since underflow of small values in
    the pdf...(Tell me how to change pdf to logpdf)
    """
    D = init.size
    samples = np.zeros((iters, D))
    y = np.zeros((k, D))
    w = np.zeros(k)
    state = init
    acceptance = 0
    for i in range(iters):
        for j in range(k):
            y[j] = np.random.multivariate_normal(np.zeros(D), sigma)
            w[j] = pdf(y[j]) / helperGaussian(y[j], sigma)
            assert not np.isnan(w[j]) and not np.isinf(w[j])
        W = sum(w)
        pickY = y[proportionProb(w)]
        if np.random.rand() < W / (W - pdf(pickY)/helperGaussian(pickY, sigma)\
                                     + pdf(state)/helperGaussian(state, sigma)):
            state = pickY
            acceptance += 1
        samples[i, :] = state
    acceptRate = (acceptance/float(iters))
    samples = samples[burnin:]
    index = np.array(range(samples.shape[0]))
    return samples[index % Mstep == 0], acceptRate


def MH_Sampling(init, log_pdf, iters, burnin, Mstep, sigma):
    """
    Basic Metropolis-Hastings sampler using random walk (gaussian proposal with
    0 mean and sigma variance).
    """
    D = init.size
    samples = np.zeros((iters, D))
    state = init
    lp = log_pdf(state)
    acceptance = 0
    #i = 0
    #total = 0
    for i in range(iters):
    #while i < iters:
        #total += 1
        prop = np.random.multivariate_normal(state, sigma)
        lpp = log_pdf(prop)
        if np.log(np.random.rand()) < lpp - lp:
            state = prop
            lp = lpp
            acceptance += 1
        samples[i, :] = state
            #i += 1
    acceptRate = (acceptance/float(iters))
    #acceptRate = (acceptance/float(total))
    samples = samples[burnin:]
    index = np.array(range(samples.shape[0]))
    return samples[index % Mstep == 0], acceptRate

def steppingOut(f, x, y, w, m):
    """
    Used in slice sampling to find a proper inital range of interest.
    """
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
    
def shrinkage(f, x, y, tupleLR):
    """
    Used in slice sampling to shrink the range and draw a sample.
    """
    l, r = tupleLR[0], tupleLR[1]
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
    """
    Basic slice sampling using step-out scheme. This sampler seems
    work better than MH in 1-dimension situation.
    Maybe the pdf can be changed to logpdf for better precision.
    """
    samples = np.zeros(iters)
    for i in range(iters):
        y = np.random.rand() * pdf(x)
        (L, R) = steppingOut(pdf, x, y, w, m)
        s = shrinkage(pdf, x, y, (L, R))
        samples[i] = s
        x = s
    return samples

def steppingOut_multi(pdf, z, index, y, w, m):
    """
    Multidimension version of steppingOut.
    """
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

def shrinkage_multi(pdf, z, index, y, tupleLR):
    """
    Multidimension version of shrinkage.
    """
    l, r = tupleLR[0], tupleLR[1]
    tmp = z.copy()
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
    """
    Multidimension versino of slice sampling.
    It's of poor performance compared to MH.
    """
    D = init.size
    state = init
    samples = np.zeros((iters, D))
    for i in range(iters):
        for j in range(D):
            y = np.random.rand() * pdf(state)
            (L, R) = steppingOut_multi(pdf, state, j, y, w, m)
            s = shrinkage_multi(pdf, state, j, y, (L, R))
            state[j] = s
        samples[i] = state
    return samples
 
def MHStep(logpdf, i, params, scala):
    state = params
    lp, mu = logpdf(state), state[i]

    state[i] = scala * np.random.randn() + mu
    if np.log(np.random.rand()) < logpdf(state) - lp:
        return state[i]
    else:
        return mu
            
def Gibbs_Sampling(init, logpdf, iters, scala):
    d = len(init)
    samples = np.zeros((iters, d))
    state = init
    acc = 0.
    for i in range(iters):
        for j in range(d):
            samples[i, j] = MHStep(logpdf, j, state, scala)
            if samples[i, j] != state[j]:
                acc += 1
            state[j] = samples[i, j]
    return samples, acc / (iters * d)
  
if __name__ == "__main__":
    init = np.array([np.random.rand()])
    #s, acc = MH_Sampling(init, log_pdf, 10000, 1000, 5, np.eye(1))
    s, acc = Gibbs_Sampling(init, log_pdf, 10000, np.eye(1))
    print(acc)
#    s = Slice_Sampling(np.random.rand(), pdf, 3000, 0.1, 10)
    n, bins, patches = plt.hist(s, 50, normed=1, facecolor='g', alpha=0.75)
    plt.axis([-1, 5, 0, 1])
    x = np.linspace(0, 4, 100)
    y = [pdf(a) for a in x]
    plt.plot(x,y,'r-')
    plt.show()
