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
from traits.api import HasTraits, Float, Integer

# label

#[offset] [type]          [value]          [description]
#0000     32 bit integer  0x00000801(2049) magic number (MSB first)
#0004     32 bit integer  60000            number of items
#0008     unsigned byte   ??               label
#0009     unsigned byte   ??               label
#........
#xxxx     unsigned byte   ??               label


with open('train-labels-idx1-ubyte','rb') as f:
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

with open('train-images-idx3-ubyte', 'rb') as f:
    magic = np.fromfile(f, '>u4', 1)[0]
    nimages = np.fromfile(f, '>u4', 1)[0]
    nrows = np.fromfile(f, '>u4', 1)[0]
    ncols = np.fromfile(f, '>u4', 1)[0]
    images = np.fromfile(f, np.ubyte).reshape((nimages, nrows*ncols))
print magic, nimages, nrows, ncols


class SGDLearner():

    def __init__(self, X_train, y_train, X_test, y_test,
                 random_state, eta0, alpha, T, update_period):
        self.eta0 = eta0
        self.alpha = alpha0

        self.T = T
        self.update_period = update_period


        self.X_train = X_train
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.rng = random_state
        #E_part = np.NaN((self.T // self.update_period, ))

        self.ntrain, self.ncoef = self.X_train.shape

        self.wts = np.zeros((ntrain + 1, self.ncoef + 1))
        self.grad = np.zeros((self.ntrain + 1, self.ncoef + 1))

    def reset(self):
        self.samples = pd.DataFrame(
            {'part': 0, 'train': 0, 'test': 0},
            index={'timesteps': np.arange(0, self.T, self.update_period)})
        #turn into dataframe?
        self.mn_grad = np.zeros((self.T // self.update_period, self.ncoef + 1))
        self.st_grad = np.zeros((self.T // self.update_period, self.ncoef + 1))

        self._iT = 0
        self.sgd = SGDClassifier(warm_start=True, warm_start=True,
                                 random_state=self.rng, fit_intercept=True)
        self.sgd.loss = "hinge"
        self.sgd.alpha = self.alpha
        self.sgd.learning_rate = 'constant'
        self.sgd.eta0 = self.eta0

    def learn(self):
        '''Train for update_period and store results'''
        ind = self.rng.randint(0, self.ntrain, self.update_period)
        self.sgd.partial_fit(self.X_train[ind, :], self.y_train[ind], [0, 1])

        self.calc_grad()
        mn = self.grad.mean(axis=0) * 1e-6
        st = self.grad.std(axis=0)*1e-6

        self.mn_grad[self._iT // self.update_period, :] = mn
        self.st_grad[self._iT // self.update_period, :] = st
        self._iT += self.update_period

        ind = self.rng.randint(0, self.ntrain, self.ntrain*0.1)
        self.samples[_iT, 'part'] = sgd.score(self.X_train[ind, :],
                                              self.y_train[ind])
        self.samples[_iT, 'train'] = sgd.score(self.X_train, self.y_train)
        self.samples[_iT, 'test'] = sgd.score(self.X_test, self.y_test)
        # restore orignal learning rate
        self.sgd.eta0 = eta0

    def calc_grad():
        # now calculate current gradient variance
        # by using very low learning rate and calculate gradients
        eta0 = self.sgd.eta0
        self.sgd.eta0 = 1e-6*eta0
        self.wts[0, :-1] = sgd.coef_
        self.wts[0, -1] = sgd.intercept_
        for im in range(self.ntrain):

            self.sgd.partial_fit(self.X_train[im, :].reshape((1, -1)),
                                 self.y_train[im].reshape((1, )), [0, 1])
            self.wts[im, :-1] = self.sgd.coef_
            self.wts[im, -1] = self.sgd.intercept_

        self.grad = np.diff(self.wts, axis=0)
