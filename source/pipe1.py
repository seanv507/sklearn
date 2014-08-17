# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 00:23:14 2014

@author: sean
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 00:06:25 2014

@author: sean
"""

import appnope


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans 
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
#from sklearn.grid_search import GridSearchCV
import mls

from sklearn.externals import joblib


import cifar
import convolve_transform 

import filters

# mac  disable MAC Mavericks AppNap which slows down ipython graphics
appnope.nope()
#%matplotlib
        
    

images, image_labels, label_names=cifar.load_cifar10()
n_patches=100

filter_height=6

filter_width=filter_height
ct=convolve_transform.ConvolveTransform(filter_height=filter_height, filter_width=filter_width)

# generate sub image data for diff sizes (ideally random sampling position)

patches=ct.generate_filter_data(images,n_patches)
patches_labels=np.repeat(image_labels,n_patches)


#plt.figure(1)
#image_learn.plot_image_label_patches(patches,patches_labels,label_names, 900)

flat=image_learn.Flatten()
flat.fit(patches)
gcn=filters.GCN(use_std=False)

def gen_filename(filename):
    data_dir='../data/filt{size}x{size}_npatch{n_patches}/'.format(
        size=filter_height,n_patches=n_patches)
    if not(os.path.exists(data_dir)):
        os.mkdir(data_dir)
    return data_dir+filename+'.pkl'
    
transform_filename='pca_gcn'
patches_filename=transform_filename + 'patches'


if os.path.isfile(gen_filename(transform_filename)):
    pca=joblib.load(gen_filename(transform_filename))
    pca_patches=joblib.load(gen_filename(patches_filename))
    pipeline=Pipeline([('flat',flat),('gcn',gcn),('pca',pca)])

else:
    pca = PCA(whiten=True)
    pipeline = Pipeline([('flat',flat),('gcn',gcn),('pca',pca)])
    pca_patches = pipeline.fit_transform(patches)
    
    joblib.dump(pca, gen_filename(transform_filename))
    joblib.dump(pca_patches, gen_filename(patches_filename))

n_clusters = 1600
rp = RandomPatches(n_clusters)
rp.fit(pca_patches)
tranform_filename='rp_'+transform_filename
patches_filename=transform_filename+'_patches'
joblib.dump(rp, transform_filename)

filter_pipe = Pipeline([('flat',flat),('gcn',gcn),('pca',pca),('rp',rp)])

ct.filter=filter_pipe
all_train=ct.transform(images)


seed=42
n_folds=3

kf=KFold(images.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
nrows=image_labels.shape[0]
class_dummy = scipy.sparse.coo_matrix((np.ones((nrows,)),
                                       (np.arange(nrows),
                                        image_labels))).tocsr()
train_index,test_index=iter(kf).next()

X_train=all_train[train_index,:]
y_train=class_dummy[train_index,:]
w=mls.mls(X_train, y_train, 1)
label_probs, label_tags=mls.multinom(X_train,w)
acc_train=1-(label_tags.squeeze()!=image_labels[train_index]).sum()/float(image_labels[train_index].shape[0])
print acc_train

label_probs_test, label_tags_test=mls.multinom(all_train[test_index,:],w)
acc_test=1-(label_tags_test.squeeze()!=image_labels[test_index]).sum()/float(image_labels[test_index].shape[0])

""" pipeline:
gcn-> each patch has mean zero, norm 1 
pca(whiten) -> each pixel has mean zero, std 1 over the patches (not within patch)

#mbkmeans=MiniBatchKMeans(n_clusters=n_elements, init='k-means++',  batch_size=1000 )
#mbkmeans.fit(pca_patches)
#mbkmeans_filename= '../data/mbkmeans.pkl'
#joblib.dump(mbkmeans,mbkmeans_filename)

# todo add clustering metric -
#import sklearn.cluster.k_means_
#lab,ine =sklearn.cluster.k_means_._labels_inertia(pca_gcn_patches,None,omp1.cluster_centers_)
#
#cluster_centers=pipeline.inverse_transform(mbkmeans.cluster_centers_)

#im_cluster_centers=image_learn.square_patches(cluster_centers)
#plt.figure(11)
#plt.imshow(im_cluster_centers.astype(np.uint8))
#plt.title('cluster_centres')
# currently not working well..


# one liner
#plt.imshow(image_learn.square_patches(pipeline.inverse_transform(mbkmeans.cluster_centers_)).astype(np.uint8))


#gcn=image_learn.GCN(use_std=True,sqrt_bias=10)
#gcn.fit(patches.reshape((patches.shape[0],-1)))
#
#pca_gcn=PCA(whiten=True)
#
#
#pipeline_gcn=Pipeline([('flat',flat),('gcn',gcn),('pca',pca_gcn)])
#
#pca_gcn_patches=pipeline_gcn.fit_transform(patches)

#omp1=OMP1(n_clusters=500, iterations=10)
#omp1.fit(pca_gcn_patches)
