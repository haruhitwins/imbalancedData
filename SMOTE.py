# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:06:01 2016

@author: zhl_m
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np

class Smote:
    def __init__(self, N='auto', k=5):
        self.N = N
        self.k = k
#        self.samples = samples

    def fit(self, samples):
        originSamples = samples.copy()
        y = samples[:, 0].flatten()
        if type(self.N) is str:
            pos = (y == 1).sum()
            neg = (y != 1).sum()
            self.N = int((neg - pos)/float(pos) * 100)
            if self.N <= 0:
                return samples
#            print 'auto N = ', self.N
            
        samples = samples[y == 1][:, 1:]
        n, d = samples.shape
        if self.N < 100:
            old_n = n
            n = int(float(self.N)/100 * old_n)
            keep = np.random.permutation(old_n)[:n]
            samples = samples[keep]
            self.N = 100

        N = int(self.N/100)
#        print "N = ", N
        synthetic = np.zeros((n * N, d))

        neighbors = NearestNeighbors(n_neighbors=self.k).fit(samples)
        index = 0
        for i in range(n):
            nnarray = neighbors.kneighbors(samples[i], return_distance=False).flatten()
            for _ in range(N):
                nn = np.random.randint(1, self.k) # Ignore the first one cause it's the sample itself.
                dif = samples[nnarray[nn]] - samples[i]
                synthetic[index] = samples[i] + np.random.rand(d) * dif
                index += 1
            
        tmp = np.hstack((np.ones(n * N).reshape(-1, 1), synthetic))
        return np.vstack((tmp, originSamples))
        
if __name__ == "__main__":
    samples = np.loadtxt('data/processed/glass1.dat',delimiter=',')
    y = samples[:,0]
    print('pos = ', (y==1).sum(), ' neg = ', (y!=1).sum())
    smote = Smote()
    after = smote.fit(samples)
    y = after[:,0]
    print('pos = ', (y==1).sum(), ' neg = ', (y!=1).sum())