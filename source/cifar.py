# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 17:55:48 2014

@author: sean
"""

import numpy as np
import cPickle


# todo check reconstruction error ( on eg biggest error)

def unpickle(file):

    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar10():
         
        
# we define here:
    dtype  = 'uint8'
    ntrain = 50000
    ntest  = 10000

    # we also expose the following details:
    img_shape = (3,32,32)
    img_size = np.prod(img_shape)
    #n_classes = 10
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog','horse','ship','truck']

    # prepare loading
    fnames = ['data_batch_%i' % i for i in range(1,6)]
    lenx = np.int(np.ceil((ntrain ) / 10000.)*10000)
    x = np.zeros((lenx,img_size), dtype=dtype)
    y = np.zeros(lenx, dtype=dtype)

    # load train data
    nloaded = 0
    dir_name='/Users/sean/projects/data/cifar10/cifar-10-batches-py/'    
    
    for i, fname in enumerate(fnames):
        data = unpickle(dir_name+fname)
        x[i*10000:(i+1)*10000, :] = data['data']
        y[i*10000:(i+1)*10000] = data['labels']
        nloaded += 10000
        if nloaded >= ntrain  + ntest: break;
    
    return x, y, label_names
        
        