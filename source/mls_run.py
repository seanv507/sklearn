# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 23:35:08 2014

@author: sean
"""
import numpy as np
import scipy as sp
import mls
X = np.loadtxt('../data/threshold_pca_image.txt')
y = np.loadtxt('../data/labels.txt')
y_mat = sp.sparse.coo_matrix((np.ones((X.shape[0], )),(np.arange(X.shape[0]), y)),shape=(X.shape[0],10))

#y_m = y_mat.todense()
w = mls.mls(X, y_mat, 1)