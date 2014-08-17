# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 21:30:00 2014

@author: sean
"""

import numpy as np
import quasirng

from image_transform import *

def _im2col_distinct(A, size):
    dy, dx = size
    assert A.shape[0] % dy == 0
    assert A.shape[1] % dx == 0

    ncol = (A.shape[0]//dy) * (A.shape[1]//dx)
    n_channels=1 if len(A)==2 else A.shape[2]

    R = np.empty((ncol, dx*dy*n_channels), dtype=A.dtype)
    k = 0
    for i in xrange(0, A.shape[0], dy):
        for j in xrange(0, A.shape[1], dx):
            if len(A.shape)==3:
                R[k, :] = A[i:i+dy, j:j+dx,:].ravel()
            else:
                R[k, :] = A[i:i+dy, j:j+dx].ravel()
            k += 1
    return R

def _im2col_sliding(A, size):
    dy, dx = size
    xsz = A.shape[1]-dx+1 #n patches x direction
    ysz = A.shape[0]-dy+1
    n_channels=1 if len(A)==2 else A.shape[2]

    R = np.empty((xsz*ysz, dx*dy*n_channels), dtype=A.dtype)

    for i in xrange(ysz):
        for j in xrange(xsz):
            if len(A.shape)==3:
                R[i*xsz+j, :] = A[i:i+dy, j:j+dx,:].ravel()
            else:
                R[i*xsz+j, :] = A[i:i+dy, j:j+dx].ravel()
    return R

def im2col(A, size, type='sliding'):
    """This function behaves similar to *im2col* in MATLAB.

    Parameters
    ----------
    A : 2-D (or 3-D - if multiple channels) ndarray
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




# define pipleine
# yaml script


'''
X consists of of array of images (n_images, n_rows, n_cols,n_channels)
'''

class ConvolveTransform (ImageTransform):
    # defaults necc for automation
    def __init__(self, height=32,width=32, channels=3,
                 filter_height=8,filter_width=8,stride=2, filter=None):

        self.filter_height=filter_height
        self.filter_width=filter_width
        self.filter=filter
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

    def transform(self, X):

        #Z = self.filter.transform(X[0, :, :, :].ravel())
        # deal with grayscale vs colour images
        #_bases = Z.shape[1]
        n_images = X.shape[0]
        # we don't know size of dictionary
        # how to determine size of output after transform ...

        # compute features for all training images
        
        for i in range(n_images):
            if (i % 1000 == 0):
                print('Extracting features: {0} / {1}\n'.format(i, n_images))

            # extract overlapping sub-patches into rows of 'patches'
            # patches are now rgb
            patches = im2col(X[i, :, :, :],
                             (self.filter_height, self.filter_width))
            # patches is now the data matrix of activations for each patch            
            patches = self.filter.transform(patches)
            if i==0:
                n_bases=patches.shape[1]
                XC = np.zeros((X.shape[0], n_bases*4))            
            
            # reshape to 2*numBases-channel image
            prows = self.height - self.filter_height + 1
            pcols = self.width - self.filter_width + 1
            patches = patches.reshape((prows, pcols, -1))
            # pool over quadrants
            halfr = np.round(prows/2.0)
            halfc = np.round(pcols/2.0)
            # sum in x,y directions: after 1st sum becomes axis 0
            q1 = patches[1:halfr, 1:halfc, :].sum(axis=0).sum(axis=0)
            q2 = patches[halfr+1:, 1:halfc, :].sum(axis=0).sum(axis=0)
            q3 = patches[1:halfr, halfc+1:, :].sum(axis=0).sum(axis=0)
            q4 = patches[halfr+1:, halfc+1:, :].sum(axis=0).sum(axis=0)

            # concatenate into feature vector
            XC[i, :] = np.hstack((q1.ravel(), q2.ravel(),
                                 q3.ravel(), q4.ravel()))

        return XC






'''
1) generate sub images for "filter"
2)

'''