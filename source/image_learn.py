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


class ConvolveTransform (ImageTransform):
    # defaults necc for automation
    def __init__(self, height=32,width=32, channels=3,
                 filter_height=8,filter_width=8,stride=2):

        self.filter_height=filter_height
        self.filter_width=filter_width
        self.stride=stride
        ImageTransform.__init__(self, height, width, channels)
        
    def generate_filter_data(self, X, n_samples):        
        """ 
        generate subimages (so number of samples ie rows increases)
        apply kmeans/sparse pca...
        store feature vectors
        
        """
        n_images = X.shape[0]

        max_filter_row=self.height - self.filter_height
        max_filter_col=self.width  - self.filter_width
        sample_coords=quasirng.halton(2, n_samples) * \
            [ max_filter_row, max_filter_col]
        sample_coords=(sample_coords+0.5).astype(np.int)
        
        sub_images=np.zeros((n_images* n_samples,
                             self.filter_height,
                             self.filter_width,
                             self.n_channels,))
        for i_image, image in enumerate(X):
            for i_sample,(row_offset,col_offset) in enumerate(sample_coords):
                sub_images[i_image*n_samples+i_sample,:,:,:] \
                          =image[ row_offset:row_offset+self.filter_height, 
                                  col_offset:col_offset+self.filter_width,:]
           
           # how to avoid duplicating data? and write to files, also so diff filters can be used           
        return sub_images
         
    def transform(self,X):
        """ convert each single image into features
        """
        strides_x=range(0,self.width-self.filter_width,self.stride)
        strides_y=range(0,self.height-self.filter_height,self.stride)
        sub_images=[]
        for image in X:
            img_rgb=image.reshape((self.height,self.width, self.n_channels))
            
            for y_offset in strides_y:
                for x_offset in strides_x:
                    sub_image=img_rgb[:, y_offset:y_offset+self.filter_height, 
                                      x_offset:x_offset+self.filter_width].ravel()
                    sub_images.append(sub_image)
    
'''
1) generate sub images for "filter"
2)


pretend filters are for template matching
how do we define templates?
incorporate tangent distance?
 - a take image and filter and find closest match
 
'''
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
    n_filters,filter_width,filter_height,n_channels=patches.shape
    n_tiles=np.int(np.ceil(np.sqrt(n_filters))) 
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



def rescale(images):
    '''  rescales rgb so that max(r,g,b)=255 and min(r,g,b)=0
    '''
    s=images.shape
    y=images.reshape((shape[0],shape[1]*shape[2],shape[3]))
    max_y=y.max(axis=1)
    min_y
    
def preprocess(X, n_components=50):
    ss=StandardScaler()
    
    pca = PCA(n_components=n_components, whiten=True)
    pip=Pipeline([('standardscaler',ss),('pCA', pca)])
    X_transform=pip.fit_transform(X)
    return X_transform,pip

def preproc_coates(patches):
    ''' first contrast enhance - remove image mean from each image [what about colour]      
        divide by stdev
        so each image has mean zero, stdev 1 across pixels, 
        then sphere ie apply pca and whiten ( ie all components )
    '''
    gcn_patches=gcn(patches,scale=1., subtract_mean=True, use_std=True,
                              sqrt_bias=10., min_divisor=1e-8)

    pca = PCA(n_components=X.shape[1], whiten=True)
    pca_gcn_patches=pca.fit_transform(gcn_patches)
    return pca_gcn_patches,pca
    
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
