# -*- coding: utf-8 -*-
"""
Created on Wed May 28 23:01:43 2014

@author: sean
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import quasirng

import pandas as pd
from sklearn.externals import joblib


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

def _im2col_distinct(A, size):
    dy, dx = size
    assert A.shape[0] % dy == 0
    assert A.shape[1] % dx == 0

    ncol = (A.shape[0]//dy) * (A.shape[1]//dx)
    R = np.empty((ncol, dx*dy), dtype=A.dtype)
    k = 0
    for i in xrange(0, A.shape[0], dy):
        for j in xrange(0, A.shape[1], dx):
            R[k, :] = A[i:i+dy, j:j+dx].ravel()
            k += 1
    return R

def _im2col_sliding(A, size):
    dy, dx = size
    xsz = A.shape[1]-dx+1
    ysz = A.shape[0]-dy+1
    R = np.empty((xsz*ysz, dx*dy), dtype=A.dtype)

    for i in xrange(ysz):
        for j in xrange(xsz):
            R[i*xsz+j, :] = A[i:i+dy, j:j+dx].ravel()
    return R

def im2col(A, size, type='sliding'):
    """This function behaves similar to *im2col* in MATLAB.

    Parameters
    ----------
    A : 2-D ndarray
        Image from which windows are obtained.
    size : 2-tuple
        Shape of each window.
    type : {'sliding', 'distinct'}, optional
        The type of the windows.

    Returns
    -------
    windows : 2-D ndarray
        The flattened windows stacked vertically.

    """

    if type == 'sliding':
        return _im2col_sliding(A, size)
    elif type == 'distinct':
        return _im2col_distinct(A, size)
    raise ValueError("invalid type of window")


class ImageTransform:
    def __init__(self,  height=32,width=32, n_channels=3):
        self.height=height
        self.width=width
        self.n_channels=n_channels
        
    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
        return { 'height' : self.height, 
        'width' : self.width,
        'n_channels': self.n_channels}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)

class SIFTTransform( ImageTransform):
    def __init__(self, height=32, width=32, n_channels=3):
        ImageTransform.__init__( height,width, n_channels)
        
    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
        
        return ImageTransform.get_params(self, deep=True)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
            ImageTransform.setattr(parameter, value)
            
    def transform(X):
     
        kp_list=[]
        n_images=X.shape[0]
        for image in X:
            img_rgb=image.reshape((ImageTransform.height,
                                   ImageTransform.width,
                                   ImageTransform.n_channels))
            img_bgr=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)            
            gray= cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
            
            sift = cv2.SIFT()

            kps = sift.detect(gray,None)
            fields={'pt_x':np.float,'pt_y':np.float,
            'angle':np.float, 'response':np.float,
            'octave':np.int, 'class_id':np.int}
            
            pd.DataFrame(zeros((len(kps),6),fields))
                
            for kp in kps:
                x,y =kp.pt
                angle=kp.angle
                response=kp.response
                octave=kp.octave
                class_id=kp.class_id
                
        return X
        

# define pipleine 
# yaml script

'''
pretend filters are for template matching
how do we define templates?
incorporate tangent distance?
 - a take image and filter and find closest match
 
'''
class GCN:
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



class Flatten:

    def fit( self, X, y=None):
        self.shape=X.shape
        return self
        
    def transform(self, X):
        return X.reshape((X.shape[0],-1))
        
    def inverse_transform(self,X):
        if len(X.shape)==1:
            X=X.reshape((1,-1))
        shape=list(self.shape)
        shape[0]=X.shape[0]
        return X.reshape(shape)

    
def square_patches(patches):
    ''' create square matrix from patches for plotting
    '''
    n_filters,filter_width,filter_height=patches.shape[:3]
    n_tiles=np.int(np.ceil(np.sqrt(n_filters)))
    
    if len(patches.shape)==3 or patches.shape[3]==1:
        n_channels=1
        image=np.zeros((filter_width*n_tiles,filter_height*n_tiles))
        patches=patches.reshape((n_filters,filter_width,filter_height)) # in case have singleton channel dimension
    else :
        image=np.zeros((filter_width*n_tiles,filter_height*n_tiles,3))

    for i_filt in range(n_filters):
        x_off=(i_filt % n_tiles)*filter_width
        y_off=np.int(i_filt/n_tiles)*filter_height

        image[x_off:x_off+filter_width,y_off:y_off+filter_height,:] \
            =patches[i_filt,:,:,:]
    return image

def plot_image_label_patches(patches,patch_labels, label_names, n_samples):
    
    for ilabel in range(10):
        plt.figure(ilabel+1)
        plt.title(label_names[ilabel])
        ind=(patch_labels==ilabel).nonzero()[0]
        samples=np.random.randint(ind.shape[0],size=n_samples)
        im_patches=patches[ind[samples],:,:,:]
        plt.imshow(square_patches(im_patches).astype(np.uint8))

def rescale_pca(mean,components, max_dev=110.0):
    
    '''  rescales rgb so that max(r,g,b)=255 and min(r,g,b)=0
    '''
    s=components.shape
    y=components.reshape((s[0],s[1]*s[2]*s[3]))
    max_y=np.abs(y).max(axis=1)
    scale=np.float(max_dev)/max_y
    comp=components*scale[:,np.newaxis,np.newaxis,np.newaxis]+mean
    return comp
    
class GCN:
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
