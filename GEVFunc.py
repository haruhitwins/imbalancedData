# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:47:26 2015

@author: Haolin
"""

import numpy as np

def logGEV(xi, v):
    x = 1. + v * xi
    assert (x >= 0).all(), "logGEV input error."
    if xi == 0:
        res = -np.exp(-v)
    else:
        res = -np.power(x, -1./xi)
    return res

def link(xi, eta):
    assert (eta > 0).all(), "link input error."
    if xi == 0:
        return -np.log(-np.log(eta))
    return (1./np.power(-np.log(eta), xi) - 1) / xi

def inverseLink(xi, v):
    x = 1. + v * xi
    assert (x >= 0).all(), "inverseLink input error."
    if xi == 0:
        res = np.exp(-np.exp(-v))
    else:
        res = np.exp(-np.power(x, -1./xi))
    return res

def GEV(xi, v):
    return inverseLink(xi, v)

def inverseLink2(xi, v):
    assert (1+v*xi >= 0).all(), "inverseLink input error."
    if xi == 0:
        res = np.exp(-np.exp(-v))
    else:
        res = np.exp(-np.power((1+v*xi), -1./xi))
    if (res == 0).any() :
        maxValue = res.max()
        res[res == 0] = maxValue + 1
        minValue = res.min()
        res[res == maxValue + 1] = minValue
    return res

def derivLink(xi, eta):
    assert (eta > 0).all(), "derivLink input error."
    res = 1./(eta*np.power(-np.log(eta), xi+1))
    res[res == np.inf] = 1e10
    res[res == -np.inf] = 1e-10
    return res

def derivInverseLink(xi, v):
    if xi == 0:
        return np.exp(-np.exp(-v) - v)
    else:
        x = 1. + v * xi
        a = np.power(x, -1./xi)
        return np.exp(-a) * a / x

def clip(xi, v):
    if xi > 0:
        v[v <= -1./xi] = -1./xi + 1e-8
    elif xi < 0:
        v[v >= -1./xi] = -1./xi - 1e-8
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    x = np.arange(-2, 2, 0.05)
    y0 = [GEV(0., v) for v in x]
    y1 = [GEV(0.5, v) for v in x]
    y2 = [GEV(-0.5, v) for v in x]
    
    plt.subplot(121)
    plt.title("Cumulative distribution function")
    plt.xlim(xmax=2)
    plt.xticks([-2,-1,0,1,2])
    l1, l2, l3 = plt.plot(x, y0, x, y1, x, y2)
    plt.setp(l1, c='r', ls='dotted', lw=2.0)
    plt.setp(l2, c='b', ls='--', lw=2.0)
    plt.setp(l3, c='g', ls='-.', lw=2.0)
    plt.grid(True)
    plt.legend([l3,l1,l2],[r'$\xi = -0.5$',r'$\xi = 0$',r'$\xi = 0.5$'],loc='upper left')
#    plt.savefig("F:/myPaper/pdf/cdf.pdf")
#    plt.show()
    
    x0 = np.arange(-3, 3, 0.05)
    x1 = np.arange(-2, 3, 0.05)
    x2 = np.arange(-3, 2, 0.05)
    y0 = [derivInverseLink(0., v) for v in x0]
    y1 = [derivInverseLink(0.5, v) for v in x1]
    y2 = [derivInverseLink(-0.5, v) for v in x2]
    
    plt.subplot(122)
    plt.title("Probability density function")
    plt.ylim(ymax=0.45)
    plt.yticks([0.05,0.15,0.25,0.35,0.45])
    l1, l2, l3 = plt.plot(x0, y0, x1, y1, x2, y2)
    plt.setp(l1, c='r', ls='dotted', lw=2.0)
    plt.setp(l2, c='b', ls='--', lw=2.0)
    plt.setp(l3, c='g', ls='-.', lw=2.0)
    plt.grid(True)
#    plt.legend([l3,l1,l2],[r'$\xi = -0.5$',r'$\xi = 0$',r'$\xi = 0.5$'],loc='upper left')
#    plt.savefig("F:/myPaper/pdf/cdfpdf.pdf", bbox_inches='tight')
    plt.show()