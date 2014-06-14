# -*- coding: utf-8 -*-
"""
Created on Wed May 28 23:01:43 2014

@author: sean
"""

import cv2
import numpy as np
import cPickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV



# todo check reconstruction error ( on eg biggest error)

def unpickle(file):

    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict



def load_cifar10():
    for  batch in range(1,2):     
        file_name='/Users/sean/projects/data/cifar10/cifar-10-batches-py/data_batch_{0}'.format(batch)    
        cifars=unpickle(file_name)    
        images=cifars['data']
        return images,cifars['labels']


class ImageTransform:
    def __init__(self, n_channels=3, height=32,width=32):
        self.n_channels=n_channels
        self.height=height
        self.width=width
        
    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
        return {'n_channels': self.n_channels, 'height' : self.height, 
        'width' : self.width}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)

class SIFTTransform( ImageTransform):
    def __init__(self, n_channels=3, height=32,width=32):
        ImageTransform.__init__(n_channels, height,width)
        
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
            img_rgb=image.reshape((self.n_channels,self.height,self.width))
            img_rgb=np.swapaxes(img_rgb,0,2)
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
    def __init__(self, channels=3,height=32,width=32, filter_height=8,filter_width=8,stride=2):

        self.filter_height=filter_height
        self.filter_width=filter_width
        self.stride=stride
        ImageTransform.__init__(self, channels, height, width)
        
    def generate_filter_data(self, X):        
        """ 
        generate subimages (so number of samples ie rows increases)
        apply kmeans/sparse pca...
        store feature vectors
        
        """        
        strides_x=range(0,self.width-self.filter_width,self.stride)
        strides_y=range(0,self.height-self.filter_height,self.stride)
        sub_images=[]
        for image in X:
            img_rgb=image.reshape((self.n_channels,self.height,self.width))
            
            for y_offset in strides_y:
                for x_offset in strides_x:
                    sub_image=img_rgb[:, y_offset:y_offset+self.filter_height, 
                                      x_offset:x_offset+self.filter_width].ravel()
                    sub_images.append(sub_image)
           
           # how to avoid duplicating data? and write to files, also so diff filters can be used           
        all_sub_images=np.vstack(sub_images)
        return all_sub_images
         
    def transform(self,X):
        """ convert each single image into features
        """
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

def gcn_covar(X,n_channels=3, scale=1., subtract_mean=True, use_std=False,
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
    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if n_pixels == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=2, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=2)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, :, np.newaxis]  # Does not make a copy.
    return X


    
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
    
    
    
#img=cv2.drawKeypoints(gray,kp)

#cv2.imwrite('sift_keypoints.png',img)
#cv2.imwrite('img.png',img_bgr)

#X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33)
#n_train=X_train.shape[0]
#kf=KFold(n=n_train, n_folds=10 )

#log_r = LogisticRegression()

#gs=GridSearchCV(log_r, param_grid={'C':[0.01,0.1,1,10,100]}, cv=kf)

# generate sub image data for diff sizes (ideally random sampling position)
# do kmeans
# recover 


