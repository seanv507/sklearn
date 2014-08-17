# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:12:40 2014

@author: sean
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
X_transform=pca.fit_transform(X)
X_inv=pca.inverse_transform(X_transform)
print 'below 2 matrices should be equal'
print X
print X_inv
print 'whiten'
pca.set_params(whiten=True)
X_transform=pca.fit_transform(X)
X_inv=pca.inverse_transform(X_transform)
print 'below 2 matrices should be equal'
print X
print X_inv

print 'whiten'
print 'pca 1 component'
pca.set_params(n_components=1)
X_transform=pca.fit_transform(X)
X_inv=pca.inverse_transform(X_transform)
print X

print 'below 2 matrices should be equal'
print X_inv
# reproject
X_transform=pca.fit_transform(X_inv)
X_inv=pca.inverse_transform(X_transform)
print X_inv
