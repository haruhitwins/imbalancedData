# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:55:32 2016

@author: zhl_m
"""
import os
from collections import Counter

class Result(object):
    def __init__(self, path):
        self.clfName = os.path.splitext(os.path.split(path)[-1])[0]
        self.scores = {}
        with open(path) as f:
            i = 0
            for l in f:
                l = l.strip().split(',')
                if len(l) == 1: 
                    self.scores[l[0].split(' ')[0]] = "PerfectSeparationError"
                    continue
                if i % 2 == 0:
                    scores = {}
                    scores['b'] = {'mean':float(l[1])}
                    scores['b1'] = {'mean':float(l[2])}
                    scores['b0'] = {'mean':float(l[3])}
                    scores['c'] = {'mean':float(l[4])}
                    scores['a'] = {'mean':float(l[5])}
                    scores['r'] = {'mean':float(l[6])}
                    scores['p'] = {'mean':float(l[7])}
                    scores['f'] = {'mean':float(l[8])}
                    scores['m'] = {'mean':float(l[9])}
                    scores['g'] = {'mean':float(l[10])}
                    self.scores[l[0]] = scores
                else:
                    self.scores[l[0]]['b']['std'] = float(l[1])
                    self.scores[l[0]]['b1']['std'] = float(l[2])
                    self.scores[l[0]]['b0']['std'] = float(l[3])
                    self.scores[l[0]]['c']['std'] = float(l[4])
                    self.scores[l[0]]['a']['std'] = float(l[5])
                    self.scores[l[0]]['r']['std'] = float(l[6])
                    self.scores[l[0]]['p']['std'] = float(l[7])
                    self.scores[l[0]]['f']['std'] = float(l[8])
                    self.scores[l[0]]['m']['std'] = float(l[9])
                    self.scores[l[0]]['g']['std'] = float(l[10])
                i += 1

def isAllValid(resList, name):
    for res in resList:
        if res.scores[name] == "PerfectSeparationError":
            return False
    return True
    
def findBest(resList, name, metric):
    if metric in ['b', 'b1', 'b0', 'c']:
        best, bestScore = 0, 1.
        for i, res in enumerate(resList):
            if res.scores[name][metric]['mean'] < bestScore:
                best, bestScore = i, res.scores[name][metric]['mean']
    else:
        best, bestScore = 0, -1.
        for i, res in enumerate(resList):
            if res.scores[name][metric]['mean'] > bestScore:
                best, bestScore = i, res.scores[name][metric]['mean']
    return resList[best].clfName

def findBestForAllData(resList, metric):
    names = resList[0].scores.keys()
    tupleList = []
    for name in names:
        if isAllValid(resList, name):
            tupleList.append((name, findBest(resList, name, metric)))
    return tupleList

def bestCountsForClfs(tupleList):
    cnt = Counter()
    for name, clf in tupleList:
        cnt[clf] += 1
    return cnt

def query(resList, name, metrics):
    for m in metrics:
        print(m+':')
        for res in resList:
            print(res.scores[name][m]['mean'])
    
if __name__ == "__main__":
    logit = Result("log/GLM_logisticRegression.csv")
    probit = Result("log/GLM_probitRegression.csv")
    cloglog = Result("log/GLM_cloglogRegression.csv")
    slogit = Result("log/SMOTE_GLM_logisticRegression.csv")
    sprobit = Result("log/SMOTE_GLM_probitRegression.csv")
    scloglog = Result("log/SMOTE_GLM_cloglogRegression.csv")
    mapgev = Result("log/MAPGEV_50iters.csv")
    mapgls = Result("log/GLS100Reg0_MAPGEV50.csv")
    mapglscv = Result("log/GLS100CVRegBrier_MAPGEV50.csv")
    mapglscv2 = Result("log/GLS100CVRegBrier_MAPGEV20.csv")
    mapglsb = Result("log/GLS100TryEachReg_MAPGEV50.csv")
    mapglsb2 = Result("log/GLS100TryEachReg_MAPGEV20.csv")
    mapglsf1 = Result("log/GLS100TryEachRegF1_MAPGEV50.csv")
    mapglsa = Result("log/GLS100TryEachRegAUC_MAPGEV50.csv")
    linsvm = Result("log/linearSVMwithPlattCalibration.csv")
    resList = [mapglscv2, logit, probit, cloglog, slogit, sprobit, scloglog]
    tupleListA = findBestForAllData(resList, 'a')
    tupleListB = findBestForAllData(resList, 'b')
    tupleListC = findBestForAllData(resList, 'c')
#    print bestCountsForClfs(tupleListC)
    for data in ['wisconsin','pima','glass0','haberman','vehicle0','segment0', \
                 'ecoli-0-3-4_vs_5','yeast-0-3-5-9_vs_7-8','vowel0', \
                 'led7digit-0-2-4-5-6-7-8-9_vs_1','page-blocks-1-3_vs_4', \
                 'flare-F','car-good','winequality-red-8_vs_6','kr-vs-k-zero_vs_eight',\
                 'poker-8_vs_6','abalone19']:
        print(scloglog.scores[data]['c']['mean'])
#    query(resList, 'vehicle0', ['a','b','c'])
#    for i in xrange(len(tupleListA)):
#        if tupleListA[i] == tupleListB[i] and tupleListB[i] == tupleListC[i]:
#            print tupleListA[i]
#    bestData = set()
#    for name, clf in tupleList:
#        if clf == mapglseach.clfName:
#            bestData.add(name)
#    with open("log/dataDescription.csv") as f:
#        for l in f:
#            name = l.strip().split(',')[0]
#            if name in bestData:
#                print l
    