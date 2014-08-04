# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 11:56:21 2014

@author: alexey
"""

import time

import pylab as pl
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import extract_patches

def gcn(data):
    data -= np.mean(data, axis=0)
    data /= np.sqrt(np.var(data, axis=0) + 10)
    return data

def whiten(data, white_alpha=0.01):
    pp = np.cov(data, rowvar=0)
    S, U = np.linalg.eigh(pp)
    S2 = 1/np.sqrt(S + white_alpha)
    S2 = np.diag(S2)
    return data.dot(U).dot(S2).dot(U.T)

def plot_gallery2(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        im = images[i].reshape((h, w, 3))
        # Normalize image to  0..1 range for visualization
        if im.dtype.name == 'float64':
            m0 = np.min(im)
            m1 = np.max(im)
            im = (im - m0) / (m1 - m0)
        pl.imshow(im)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


def recheck2(patches, ordered=True, psize=32):
    if ordered:
        ixs = range(16)
    else:
        ixs = numpy.random.choice(range(patches.shape[0]), size=16, replace=False)
    plot_gallery(patches[ixs], range(16), psize, psize, 4, 4)


class MiniCluster:
    def __init__(self, nclusters=1000, psize=16):
        self.psize = 16
        self.patch_size = (self.psize, self.psize)
        self.nclusters = nclusters
        self.rng = np.random.RandomState(0)
        self.kmeans = MiniBatchKMeans(n_clusters=nclusters, random_state=self.rng, verbose=True)
        
    def fit(self, images):
        buffer = []
        index = 1
        t0 = time.time()

        # The online learning part: cycle over the whole dataset 4 times
        index = 0
        passes = 10
        for _ in range(passes):
            for img in images:
                data = extract_patches_2d(img, self.patch_size, max_patches=15,
                                          random_state=self.rng)
                data = np.reshape(data, (len(data), -1))
                #This casting is only needed for RGB data
                #buffer.append(data.astype(float))
                buffer.append(data)
                index += 1
                #if index % 1000 == 0:
                if index % (self.nclusters * 2) == 0:
                    data = np.concatenate(buffer, axis=0)
                    data = gcn(data)
                    data = whiten(data)
                    self.kmeans.partial_fit(data)
                    buffer = []
                          
        dt = time.time() - t0
        print('done in %.2fs.' % dt)

def image_conv_patches(image):
    # Convolution patches
    patches = extract_patches(image, 16)
    w, h = patches.shape[0], patches.shape[1]
    patches = patches.reshape((w*h,16 * 16))
    # GCN
    patches = patches - patches.mean(axis=1)[:, np.newaxis]
    patches /= np.sqrt(patches.var(axis=1) + 10)[:, np.newaxis]
    return (w, h, patches)

def dict_threshold(alpha,dictionary,patches):
    patches = dictionary.dot(patches.transpose()).transpose()
    patches = np.maximum(0, patches - alpha)
    return patches
    
def featurize_image(shrink, alpha, dictionary, image):
    w, h, patches = image_conv_patches(image)
    nclusters = dictionary.shape[0]
    patches = dict_threshold(alpha, dictionary, patches)
    # Now, perform pooling (with averaging)
    patches = patches.reshape((w, h, -1))
    wr = int(w / shrink)
    hr = int(h / shrink)
    step = min(wr-1,hr-1)
    patches = extract_patches(patches,patch_shape=(wr,hr,nclusters),extraction_step=step)
    nw, nh = patches.shape[0], patches.shape[1]
    patches = patches.reshape((nw,nh,wr,hr,nclusters))
    patches = patches.mean(axis=(2,3))
    patches = patches.reshape((nw*nh*nclusters))
    return patches
    
def featurize(shrink, alpha, dictionary, images):
    new_x = []    
    for ix in range(images.shape[0]):
        new_x = np.append(new_x, featurize_image(shrink, alpha, dictionary, images[ix]))
    return np.reshape(new_x, newshape=(images.shape[0],-1))
    
class Sparsify:
    def __init__(self, nclusters=1000, psize=16, alpha=0.25, shrink=3):
        self.mc = MiniCluster(nclusters, psize)
        self.psize = psize
        self.alpha = alpha
        self.shrink = shrink
        
    def fit(self, images):
        self.mc.fit(images)
        return self.mc.kmeans.cluster_centers_
        
    def transform(self, images):
        return featurize(self.shrink, self.alpha, self.mc.kmeans.cluster_centers_, images)
        