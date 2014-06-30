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
#from sklearn.cross_validation import KFold
#from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib


import cifar
import image_learn

# mac  disable MAC Mavericks AppNap which slows down ipython graphics
appnope.nope()
#%matplotlib
        
    

images, image_labels, label_names=cifar.load_cifar10()

ct=image_learn.ConvolveTransform()
# generate sub image data for diff sizes (ideally random sampling position)
patches=ct.generate_filter_data(images,20)
patches_labels=np.repeat(image_labels,20)


plt.figure(1)
image_learn.plot_image_label_patches(patches,patches_labels,label_names, 900)

flat=image_learn.Flatten()
flat.fit(patches)

pca_filename='../data/pca.pkl'
pca_patches_filename='../data/pca_patches.pkl'

if os.path.isfile(pca_filename):
    pca=joblib.load(pca_filename)
    pca_patches=joblib.load(pca_patches_filename)
    pipeline=Pipeline([('flat',flat),('pca',pca)])

else:
    pca=PCA(whiten=True)
    pipeline=Pipeline([('flat',flat),('pca',pca)])
    pca_patches=pipeline.fit_transform(patches)
    
    joblib.dump(pca,pca_filename)
    joblib.dump(pca,pca_patches_filename)
    

n_clusters=500

mbkmeans=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',  batch_size=1000 )
#mbkmeans.fit(pca_patches)
mbkmeans_filename= '../data/mbkmeans.pkl'
#joblib.dump(mbkmeans,mbkmeans_filename)

#cluster_centers=pipeline.inverse_transform(mbkmeans.cluster_centers_)

#im_cluster_centers=image_learn.square_patches(cluster_centers)
#plt.figure(11)
#plt.imshow(im_cluster_centers.astype(np.uint8))
#plt.title('cluster_centres')
# currently not working well..


# one liner
#plt.imshow(image_learn.square_patches(pipeline.inverse_transform(mbkmeans.cluster_centers_)).astype(np.uint8))


gcn=image_learn.GCN(use_std=True,sqrt_bias=10)
gcn.fit(patches.reshape((patches.shape[0],-1)))

pipeline_gcn=Pipeline([('flat',flat),('gcn',gcn),('pca',pca)])

pca_gcn_patches=pipeline_gcn.transform(patches)

omp1=OMP1(n_clusters=500, iterations=10)
omp1.fit(pca_gcn_patches)


#to do add clustering metric -
import sklearn.cluster.k_means_
lab,ine =sklearn.cluster.k_means_._labels_inertia(pca_gcn_patches,None,omp1.cluster_centers_)








