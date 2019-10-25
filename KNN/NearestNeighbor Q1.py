# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:41:10 2017

"""
import numpy as np
from collections import Counter

class NearestNeighbor(object):
    #http://cs231n.github.io/classification/
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k, l='L1'):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            
            neighbors = []
            min_index = np.argpartition(distances,k) # get the index with smallest distance
           
            for z in range(k):
                neighbors.append(min_index[z])
           
            common= Counter(self.ytr[neighbors]).most_common(1)[0]
            Ypred[i] = common[0] # predict the label of the nearest example        
       
        return Ypred
