# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 23:15:09 2014

@author: sean
"""

import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

def load_data():
    # label

    #[offset] [type]          [value]          [description]
    #0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    #0004     32 bit integer  60000            number of items
    #0008     unsigned byte   ??               label
    #0009     unsigned byte   ??               label
    #........
    #xxxx     unsigned byte   ??               label

    with open('../data/train-labels-idx1-ubyte','rb') as f:
        magic = np.fromfile(f,'>u4',1)[0]
        nlabels = np.fromfile(f,'>u4',1)[0]
        labels = np.fromfile(f,np.ubyte)
    print magic, nlabels
    
    
    #TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    #
    #[offset] [type]          [value]          [description]
    #0000     32 bit integer  0x00000803(2051) magic number
    #0004     32 bit integer  60000            number of images
    #0008     32 bit integer  28               number of rows
    #0012     32 bit integer  28               number of columns
    #0016     unsigned byte   ??               pixel
    #0017     unsigned byte   ??               pixel
    #........
    #xxxx     unsigned byte   ??               pixel
    # Pixels are organized row-wise. Pixel values are 0 to 255.
    # 0 means background (white), 255 means foreground (black).
    
    with open('../data/train-images-idx3-ubyte', 'rb') as f:
        magic = np.fromfile(f, '>u4', 1)[0]
        nimages = np.fromfile(f, '>u4', 1)[0]
        nrows = np.fromfile(f, '>u4', 1)[0]
        ncols = np.fromfile(f, '>u4', 1)[0]
        images = np.fromfile(f, np.ubyte).reshape((nimages, nrows*ncols))
    print magic, nimages, nrows, ncols
    return images,labels,nimages,nrows,ncols

class SGDLearner():

    def __init__(self, X_train, y_train, X_test, y_test,
                 random_state, eta0, alpha, T, update_period):
        self.eta0 = eta0
        self.alpha = alpha

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.rng = random_state
        #E_part = np.NaN((self.T // self.update_period, ))

        self.ntrain, self.ncoef = self.X_train.shape

        self.wts = np.zeros((self.ntrain + 1, self.ncoef + 1))
        self.grad = np.zeros((self.ntrain + 1, self.ncoef + 1))
        self.reset()

    def reset(self):
        
        self.scores=[]
        
        #turn into dataframe?
        self.mn_grad = []
        self.st_grad = []

        self._iT = 0
        self.sgd = SGDClassifier(warm_start=True,
                                 random_state=self.rng, fit_intercept=True)
        self.sgd.loss = "hinge"
        self.sgd.alpha = self.alpha
        self.sgd.learning_rate = 'constant'
        self.sgd.eta0 = self.eta0

    def learn(self, nsamples):
        '''Train for nsamples and store results'''
        ind = self.rng.randint(0, self.ntrain, nsamples)
        self.sgd.partial_fit(self.X_train[ind, :], self.y_train[ind], [0, 1])
        self._iT += nsamples

        self.calc_grad()
        mn = self.grad.mean(axis=0) * 1e-6
        st = self.grad.std(axis=0)*1e-6

        self.mn_grad.append(mn)
        self.mn_grad.append(st)

        ind = self.rng.randint(0, self.ntrain, int(self.ntrain*0.1))
        scores = {'timestep': self._iT}
        scores['part'] = \
            self.sgd.score(self.X_train[ind, :], self.y_train[ind])
        scores['train'] = \
            self.sgd.score(self.X_train, self.y_train)
        scores['test'] = \
            self.sgd.score(self.X_test, self.y_test)
        self.scores.append(scores)

    def calc_grad(self):
        # now calculate current gradient variance
        # by using very low learning rate and calculate gradients
        eta0 = self.sgd.eta0
        self.sgd.eta0 = 1e-6*eta0
        self.wts[0, :-1] = self.sgd.coef_
        self.wts[0, -1] = self.sgd.intercept_
        for im in range(self.ntrain):

            self.sgd.partial_fit(self.X_train[im, :].reshape((1, -1)),
                                 self.y_train[im].reshape((1, )), [0, 1])
            self.wts[im, :-1] = self.sgd.coef_
            self.wts[im, -1] = self.sgd.intercept_

        self.grad = np.diff(self.wts, axis=0)
        self.sgd.eta0 = eta0
        # restore orignal learning rate


#labels8=labels==8
#
## split into train and test set
#
#images_all=np.zeros((nimages,nrows*ncols))
#images_all[:]=images[:]
#labels_all=labels8
#
#
## split into train and test set
#rng=np.random.RandomState(2)
#images_train,images_test, labels_train,labels_test = train_test_split(images_all, labels_all,test_size=0.33, random_state=rng)
#ntrain=images_train.shape[0]
#
#images_train_mean=images_train.mean()
#images_train_std=images_train.std()
#
## rescale
#scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
#images_train_scale=scaler.fit_transform(images_train)
#images_test_scale=scaler.transform(images_test)
