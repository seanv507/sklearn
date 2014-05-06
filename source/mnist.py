# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 23:15:09 2014

@author: sean
"""

import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# label

#[offset] [type]          [value]          [description] 
#0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
#0004     32 bit integer  60000            number of items 
#0008     unsigned byte   ??               label 
#0009     unsigned byte   ??               label 
#........ 
#xxxx     unsigned byte   ??               label


#with open('train-labels-idx1-ubyte','rb') as f:
#    magic=np.fromfile(f,'>u4',1)[0]
#    nlabels=np.fromfile(f,'>u4',1)[0]
#    labels=np.fromfile(f,np.ubyte)
#print magic,nlabels
#    

#TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
#
#[offset] [type]          [value]          [description] 
#0000     32 bit integer  0x00000803(2051) magic number 
#0004     32 bit integer  60000            number of images 
#0008     32 bit integer  28               number of rows 
#0012     32 bit integer  28               number of columns 
#0016     unsigned byte   ??               pixel 
#0017     unsigned byte   ??               pixel 
#........ 
#xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255.
# 0 means background (white), 255 means foreground (black).

#with open('train-images-idx3-ubyte','rb') as f:
#    magic=np.fromfile(f,'>u4',1)[0]
#    nimages=np.fromfile(f,'>u4',1)[0]
#    nrows=np.fromfile(f,'>u4',1)[0]
#    ncols=np.fromfile(f,'>u4',1)[0]
#    images=np.fromfile(f,np.ubyte).reshape((nimages,nrows*ncols))
#print magic,nimages,nrows,ncols

labels9=labels==9

# split into train and test set

images_all=np.zeros((nimages,nrows*ncols))
images_all[:]=images[:]
labels_all=labels9


# split into train and test set
rng=np.random.RandomState(2)
images_train,images_test, labels_train,labels_test = train_test_split(images_all, labels_all,test_size=0.33, random_state=rng)
ntrain=images_train.shape[0]

images_train_mean=images_train.mean()
images_train_std=images_train.std()

# rescale
scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
images_train_scale=scaler.fit_transform(images_train)
images_test_scale=scaler.transform(images_test)
#train for a bit
# fit the model



n_epochs=10000
mini=100
eta0=0.01
sgd = SGDClassifier(loss="hinge", alpha=0.01, n_iter=1, fit_intercept=True,
                    learning_rate='constant',eta0=eta0,warm_start=True,random_state=rng)
                    
E=pd.DataFrame(np.zeros((n_epochs,3)),columns=['part','all','test'])
wts=np.zeros((ntrain+1,nrows*ncols+1))
grad=np.zeros((ntrain+1,nrows*ncols+1))
mn_grad=np.zeros((n_epochs,nrows*ncols+1))
st_grad=np.zeros((n_epochs,nrows*ncols+1))

for i in range(n_epochs):
    #  train for a bit
    ind=rng.randint(0,ntrain,mini)
    sgd.partial_fit(images_train[ind,:], labels_train[ind],[0,1])
        # then set     
    eta0=sgd.eta0
    sgd.eta0=1e-6
    wts[0,:-1]=sgd.coef_
    wts[0,-1]=sgd.intercept_
    for im in range(ntrain):
    
        sgd.partial_fit(images_train[im,:].reshape((1,-1)), labels_train[im].reshape((1,)),[0,1])
        wts[im,:-1]=sgd.coef_
        wts[im,-1]=sgd.intercept_
    grad=np.diff(wts,axis=0)
    mn=grad.mean(axis=0)*1e-6
    st=grad.std(axis=0)*1e-6
    ind=rng.randint(0,ntrain,mini)
    mn_grad[i,:]=mn
    st_grad[i,:]=st
    E["part"][i]=sgd.score(images_train_scale[ind,:], labels_train[ind])
    E["all"][i]=sgd.score(images_train_scale, labels_train)
    E["test"][i]=sgd.score(images_test_scale, labels_test)
    
    sgd.eta0=eta0
