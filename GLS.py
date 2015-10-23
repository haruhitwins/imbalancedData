# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:58:37 2015

@author: Haolin
"""

import numpy as np

def inverseLink(xi, v):
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

def derivInverseLink(xi, v):
    if xi == 0:
        return np.exp(-np.exp(-v) - v)
    else:
        return np.exp(-np.power((1+v*xi), -1./xi)) \
               * np.power(1+v*xi, -1./xi - 1)
    
def calculateL(xi):
    if xi == 0: return 1 / np.e
    if xi > -1 : 
        v = (np.power(1./(xi + 1), xi) - 1)/xi
    if xi <= -1:
        v = -1./xi
    print 'v = ', v
    return derivInverseLink(xi, v)
    
xis = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
for xi in xis:
    print calculateL(xi)