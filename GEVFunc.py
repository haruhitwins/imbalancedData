# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:47:26 2015

@author: Haolin
"""

import numpy as np

def link(xi, eta):
    assert (eta > 0).all(), "link input error."
    if xi == 0:
        return -np.log(-np.log(eta))
    return (1./np.power(-np.log(eta), xi) - 1) / xi

def inverseLink(xi, v):
    assert (1+v*xi >= 0).all(), "inverseLink input error."
    if xi == 0:
        res = np.exp(-np.exp(-v))
    else:
        res = np.exp(-np.power((1+v*xi), -1./xi))
    return res

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
        return np.exp(-np.power((1+v*xi), -1./xi)) \
               * np.power(1+v*xi, -1./xi - 1)

def clip(xi, v):
    if xi > 0:
        v[v < -1./xi] = -1./xi
    elif xi < 0:
        v[v > -1./xi] = -1./xi