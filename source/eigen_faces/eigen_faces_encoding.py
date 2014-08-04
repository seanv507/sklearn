# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 03:01:34 2014

@author: alexey
From: http://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html#example-cluster-plot-dict-face-patches-py

Online learning of a dictionary of parts of faces
==================================================

This example uses a large dataset of faces to learn a set of 20 x 20
images patches that constitute faces.

From the programming standpoint, it is interesting because it shows how
to use the online API of the scikit-learn to process a very large
dataset by chunks. The way we proceed is that we load an image at a time
and extract randomly 15 patches from this image. Once we have accumulated
750 of these patches (using 50 images), we run the `partial_fit` method
of the online KMeans object, MiniBatchKMeans.

The verbose setting on the MiniBatchKMeans enables us to see that some
clusters are reassigned during the successive calls to
partial-fit. This is because the number of patches that they represent
has become too low, and it is better to choose a random new
cluster.
"""

print(__doc__)

import time

import pylab as pl
import numpy as np


from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




import mini_batch_clusters

# Requirements, easy to connect subsequent stages
# Easy to plot intermediate data
# Easy to explore hyper-parameters for improving performance

#faces = datasets.fetch_olivetti_faces()
faces = fetch_lfw_people(min_faces_per_person=70)
target_names = faces.target_names
n_classes = target_names.shape[0]



###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(faces.images, faces.target, test_size=0.25)



# Show random selection
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

def recheck(patches, ordered=True):
    if ordered:
        ixs = range(16)
    else:
        ixs = numpy.random.choice(range(patches.shape[0]), size=16, replace=False)
    plot_gallery(patches[ixs], range(16), psize, psize, 4, 4)

###############################################################################
# Learn the dictionary of images

print('Learning the dictionary... ')
sp = mini_batch_clusters.Sparsify()
sp.fit(X_train)

###############################################################################
# Plot the results
cc = np.random.choice(sp.mc.kmeans.labels_,size=81)
cc = sp.mc.kmeans.cluster_centers_[cc]
pl.figure(figsize=(4.2, 4))
for i, patch in enumerate(cc):
    pl.subplot(9, 9, i + 1)
    pl.imshow(patch.reshape(patch_size), cmap=pl.cm.gray,
              interpolation='nearest')
    pl.xticks(())
    pl.yticks(())


pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

pl.show()

pl.hist(sp.mc.kmeans.labels_,nclusters)
pl.show()

recheck(cc)


np.convolve

###############################################################################
# Generate sparse features
X_train_sparse = sp.transform(X_train)
X_test_sparse = sp.transform(X_test)

###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time.time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}
clf = GridSearchCV(LogisticRegression(class_weight='auto'), param_grid)

clf = clf.fit(X_train_sparse, y_train)
print("done in %0.3fs" % (time.time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time.time()
y_pred = clf.predict(X_test_sparse)
print("done in %0.3fs" % (time.time() - t0))

print("Accuracy score: %f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# Precision / Recall for C=10000.0: 0.95 / 0.95 (Accuracy 0.96)
# Versus 0.8 / 0.8 for eigen faces (Accuracy 0.8)
