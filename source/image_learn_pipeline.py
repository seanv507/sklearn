# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 00:06:25 2014

@author: sean
"""
import cifar10
import image_learn
import joblib


        
    

images, image_labels, label_names=load_cifar10()

ct=ConvolveTransform()
# generate sub image data for diff sizes (ideally random sampling position)
patches=ct.generate_filter_data(images,20)
patches_labels=np.repeat(image_labels,20)


plt.figure(1)
plot_image_label_patches(patches,patches_labels, 900)

flat=Flatten()
pca=PCA(whiten=True)

pipeline=Pipeline([('flat',flat),('pca',pca)])
pca_patches=pipeline.fit_transform(patches)
joblib.dump('../data/pca.pkl')
joblib.dump('../data/pca_patches.pkl')

n_clusters=500
mbkmeans=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',  batch_size=100 )
mbkmeans.fit(pca_patches)
joblib.dump('../data/mbkmeans.pkl')

cluster_centers=pipeline.inverse_transform(mbkmeans.cluster_centers_)

im_cluster_centers=square_patches(cluster_centers)
plt.figure(11)
plt.imshow(im_cluster_centers.astype(np.uint8))
plt.title('cluster_centres')
# currently not working well..


# one liner
plt.imshow(plot_patches(pipeline.inverse_transform(mbkmeans.cluster_centers_)).astype(np.uint8))




