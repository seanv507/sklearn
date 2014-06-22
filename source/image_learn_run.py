# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 17:39:05 2014

@author: sean
"""

#img=cv2.drawKeypoints(gray,kp)

#cv2.imwrite('sift_keypoints.png',img)
#cv2.imwrite('img.png',img_bgr)

#X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33)
#n_train=X_train.shape[0]
#kf=KFold(n=n_train, n_folds=10 )

#log_r = LogisticRegression()

#gs=GridSearchCV(log_r, param_grid={'C':[0.01,0.1,1,10,100]}, cv=kf)


#precompute_distances=False
# n_clusters
#filt=KMeans(n_clusters=n_clusters, precompute_distances=False)

#filt_50000=KMeans(n_clusters=n_clusters, precompute_distances=False)
#CPU times: user 34min 3s, sys: 44.3 s, total: 34min 47s
#Wall time: 1h 33min 59s

#filt_250000=KMeans(n_clusters=n_clusters, precompute_distances=False)
#CPU times: user 5h 51min 2s, sys: 9min 26s, total: 6h 29s

#filt_1000000=KMeans(n_clusters=n_clusters, precompute_distances=False)

#filt_50000_mbatch=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',  batch_size=1000 )
filt_250000_mbatch=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',  batch_size=1000 )
filt_all_mbatch=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',  batch_size=1000 )
#
#%time filt_50000.fit(patches_whiten[0:50000,:])
#%time filt_250000.fit(patches_whiten[0:250000,:])
#%time filt_1000000.fit(patches_whiten[0:1000000,:])



%time filt_50000_mbatch.fit(patches_whiten[0:50000,:])
    #CPU times: user 6.89 s, sys: 432 ms, total: 7.32 s
    #Wall time: 5.69 s
    #Out[9]: 
    #MiniBatchKMeans(batch_size=1000, compute_labels=True, init='k-means++',
    #        init_size=None, max_iter=100, max_no_improvement=10,
    #        n_clusters=400, n_init=3, random_state=None,
    #        reassignment_ratio=0.01, tol=0.0, verbose=0)
    #
    #filt_250000_mbatch=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',  batch_size=1000 )
    #
    #%time filt_250000_mbatch.fit(patches_whiten[0:250000,:])
    #CPU times: user 10.6 s, sys: 1.18 s, total: 11.8 s
    #Wall time: 9.66 s
    #Out[11]: 
    #MiniBatchKMeans(batch_size=1000, compute_labels=True, init='k-means++',
    #        init_size=None, max_iter=100, max_no_improvement=10,
    #        n_clusters=400, n_init=3, random_state=None,
    #        reassignment_ratio=0.01, tol=0.0, verbose=0)
    #
    #%time filt_all_mbatch.fit(patches_whiten)
    #CPU times: user 49.9 s, sys: 42.7 s, total: 1min 32s
    #Wall time: 1min 19s
    #Out[13]: 
    #MiniBatchKMeans(batch_size=1000, compute_labels=True, init='k-means++',
    #        init_size=None, max_iter=100, max_no_improvement=10,
    #        n_clusters=400, n_init=3, random_state=None,
    #        reassignment_ratio=0.01, tol=0.0, verbose=0)

images, image_labels, label_names=load_cifar10()

ct=ConvolveTransform()
# generate sub image data for diff sizes (ideally random sampling position)
patches=ct.generate_filter_data(images,20)
patches_labels=np.repeat(image_labels,20)

flat=Flatten()
pca=PCA(whiten=True)
pipeline=Pipeline([('flat',flat),('pca',pca)])
pca_patches=pipeline.fit_transform(patches)


n_clusters=500
mbkmeans=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',  batch_size=1000 )
mbkmeans.fit(pca_patches)

%matplotlib
import appnope
appnope.nope()
plt.plot(mbkmeans_pca_patches[0,:],'o')
plt.plot(mbkmeans_pca_patches,'o',color=patches_labels)
plt.plot(mbkmeans_pca_patches[0:5,:],'o',color=patches_labels)
