# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 13:41:46 2014

@author: sean
"""

import numpy as np
import scipy.sparse

'''
pretend filters are for template matching
how do we define templates?
incorporate tangent distance?
 - a take image and filter and find closest match
'''

class Flatten:    

    def fit(self, X, y=None):
        self.shape = X.shape
        return self

    def transform(self, X):
        return X.reshape((X.shape[0],-1))

    def inverse_transform(self,X):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        shape = list(self.shape)
        shape[0] = X.shape[0]
        return X.reshape(shape)

class GCN:
    """ subtracts pixel mean and divides by pixel std dev (for each image).
    
    so by default transformed image has vector norm $\sqrt(n_pixels)$
    """
    def __init__(self, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
        self.scale=scale
        self.subtract_mean=subtract_mean
        self.use_std=use_std
        self.sqrt_bias =sqrt_bias
        self.min_divisor=min_divisor
    
    def fit(self, X,y=None):
        # only used for inverse transform, not dealing with use_std issues
        self.mean_=X.mean() if self.subtract_mean else 0
        assert(self.subtract_mean==True)
        self.norm_=np.sqrt(X.var(axis=1,ddof=1)+self.sqrt_bias).mean()/self.scale
        return self
    
    def transform(self,X):
        mean = X.mean(axis=1)
        if self.subtract_mean:
            X = X - mean[:, np.newaxis]  # Makes a copy.
        else:
            X = X.copy()    
             # how to deal with colour!!!
        if self.use_std:
            # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
            # Coates' code does.
            ddof = 1
    
            # If we don't do this, X.var will return nan.
            if X.shape[1] == 1:
                ddof = 0
            
            normalizers = np.sqrt(self.sqrt_bias + X.var(axis=1, ddof=ddof)) / self.scale
        else:
            normalizers = np.sqrt(self.sqrt_bias + (X ** 2).sum(axis=1)) / self.scale
    
        # Don't normalize by anything too small.
        normalizers[normalizers < self.min_divisor] = 1.
    
        X /= normalizers[:, np.newaxis]  # Does not make a copy.
        return X
        
    def inverse_transform(self, X):
        ''' Guess transform by scaling by 128 and adding 128
        '''
        return X*self.norm_+self.mean_

def gcn_covar(X,n_channels=3, scale=1., subtract_mean=True, 
                              sqrt_bias=0., min_divisor=1e-8):
    n_pixels=X.shape[1]/n_channels
    X=X.reshape((X.shape[0],n_channels,-1))
    # we standardise all images - making results "invariant" 
    # to lighting changes across images. Mean is overall colour, 
    # covar is contrast. Correl?
    mean = X.mean(axis=2)
    if subtract_mean:
        X = X - mean[:,:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()    
         # how to deal with colour!!!
    
    normalizers = np.sqrt(sqrt_bias + X.var(axis=2, ddof=ddof)) / scale
    
    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, :, np.newaxis]  # Does not make a copy.
    return X


class Threshold:
    """ return thresholded filter X.(X>alpha) (and ) X.(X < -alpha)

    coates uses max(X-alpha,0)
    if elements of X negative also return negative threshold
    """
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.atleast_2d(X)
        n_data, n_features = X.shape
        if (X<0).sum()>0:
            Z = np.zeros((n_data, 2*n_features))
            Z[:, n_features:2*n_features] = (X < -self.alpha) * X
            Z[:, 0:n_features] = (X > self.alpha) * X
        else:
            Z = (X > self.alpha) * X
        return Z


class RandomPatches:
    """Select n_clusters at random from input data and use as dictionary elements. 

    Attributes:
        n_clusters: number of patches to select for dictionary
        i_elements_: index of patches selected in original data
        cluster_centers_: (n_centers, ...) each row is dictionary elemnet
    TODO add seed method
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        n_patches=X.shape[0]
        self.i_elements_=np.sort(np.random.permutation(n_patches)[0:self.n_clusters])
        normalizers=(X[self.i_elements_,:]**2).sum(axis=1)
        self.cluster_centers_=X[self.i_elements_,:]/normalizers[:,np.newaxis]
    
    def transform(self, X):
        """ dot product with selected patches."""
        output=np.dot(X,self.cluster_centers_.T)
        return output

# to do add metrics, and verbose flag...

class OMP1:
    def __init__(self, n_clusters=None, iterations=None, batch_size=100):
        self.n_clusters=n_clusters
        self.iterations=iterations
        self.batch_size=batch_size
    
    def  gsvq_step(self, X):
    
        summation = np.zeros(self.cluster_centers_.shape)
        counts = np.zeros((self.n_clusters,1))
        self.inertia_=0        
        
        for i_start in range(0,X.shape[0],self.batch_size):
            # deal with 
            i_end=min(i_start+self.batch_size, X.shape[0]);
            act_batch_size = i_end-i_start;

            dots = self.transform(X[i_start:i_end,:]).T
            # nclusters by ndata 


            labels = np.abs(dots).argmax(axis=0) # max cluster
            # turn labels into 1-hot encoding

            E = scipy.sparse.csr_matrix((np.ones((act_batch_size,)),
                        np.vstack((labels,np.arange(act_batch_size)))),
                        shape=(self.n_clusters,act_batch_size))
                         
            
            counts += E.sum(axis=1)
            
            dots =  E.multiply(dots) # dots is now sparse
            # inertia = x^2  -2x.y +  y^2, so 
            # 2(1-abs(dots)) (ASSUMING input data (and clusters) have unit norm)
            # 
            self.inertia_-=np.abs(dots).sum()
            summation += np.dot(dots , X[i_start:i_end,:]) 
            # take sum of X vectors assigned to each cluster, weighted by dot product
            # nb weighting ensures +/- direction handled properly
        self.inertia_=2*(X.shape[0]-self.inertia_)
        # inertia will not sum over partial fits
        return summation, counts
    
        
    def partial_fit(self,X, y=None):
        self.inertia_=0
        for itr in range(self.iterations):
            print('Running GSVQ:  iteration={0}...inertia={1}\n'.
                format(itr, self.inertia_))
            
            # do assignment + accumulation
            summation,counts = self.gsvq_step(X)
        
            # reinit empty clusters
            I=np.where((summation**2).sum(axis=1) < 0.001)[0]
            summation[I,:] = np.random.randn(I.shape[0], X.shape[1])
        
            # normalize
            self.cluster_centers_ = summation/ \
                np.sqrt((summation**2).sum(axis=1)[:,np.newaxis]+1e-20)
        
        return self
    
    def fit(self, X, y=None):
        ''' finds clusters on unit ball using cosine similarity. 
            rows of X should have l2 norm 1
        ''' 
        self.cluster_centers_=np.random.randn((self.n_clusters),X.shape[1])
        self.cluster_centers_ /= \
            np.sqrt((self.cluster_centers_**2).sum(axis=1)[:,np.newaxis]+1e-20)
        
        return self.partial_fit(X,y)
  
    
    def transform(self, X):
        return np.dot(X,self.cluster_centers_.T)

  
    