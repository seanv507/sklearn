# -*- coding: utf-8 -*-
"""
Created on Wed May 28 23:01:43 2014

@author: sean
"""

import cv2
import numpy as np
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
class Flatten

def reconstruct_kmeans(pca,filt):
    n_channels=3
    filter_height,filter_width=(8,8)
    n_filters=filt.cluster_centers_.shape[0]
    filters=filt.cluster_centers_.copy()
    patches=pca.inverse_transform(filters)
    patches=patches.reshape(patches.shape[0],filter_height,filter_width,n_channels)
    # matplot lib expects  x,y, colour 

def plot_pca(components,width,height,channels):
    comp=components.copy().reshape((-1,width,height,channels))
    return comp
    
    
def plot_patches(patches):
    n_filters,filter_width,filter_height,n_channels=patches.shape
    n_tiles=np.int(np.sqrt(n_filters)+0.5) #ceiling
    image=np.zeros((filter_width*n_tiles,filter_height*n_tiles,3))

    for i_filt in range(n_filters):
        x_off=(i_filt % n_tiles)*filter_width
        y_off=np.int(i_filt/n_tiles)*filter_height

        image[x_off:x_off+filter_width,y_off:y_off+filter_height,:] \
            =patches[i_filt,:,:,:]
    return image
    
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
    

    
def gcn(X,scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()    
         # how to deal with colour!!!
    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X

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
