# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 14:39:00 2014

@author: sean
"""

import numpy as np

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
