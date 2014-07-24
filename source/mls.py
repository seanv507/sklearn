# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 20:22:22 2014

@author: sean
"""

import numpy as np
import scipy as sp


def test_data(nsamples):
    means = [[0,0], [2,2],[3,3]]
    covars = [np.eye(2),np.eye(2),np.eye(2)]
    all_data = []
    all_labels = []
    nclasses = len(means)
    for iclass, (mean, covar) in enumerate(zip(means, covars)):
        data = np.random.multivariate_normal(mean, covar, nsamples)
        label = np.zeros((1, nclasses))
        label[0, iclass] = 1
        labels = np.tile(label, (nsamples, 1))
        all_data.append(data)
        all_labels.append(labels)
    X = np.concatenate(all_data)
    X = np.hstack((X, np.ones((nclasses * nsamples, 1))))
    y = np.concatenate(all_labels)
    return X, y


def multinom(X, W):
    k = W.shape[1]
    pt = np.dot(X, W) #matrix n x k
    max_pt = pt.max(axis=1).reshape(-1, 1)
    # numerically stable exp, remove exp(max_pt) from top and bottom
    q = np.exp(pt - np.tile(max_pt, (1, k)))
    pp = q / np.tile(q.sum(axis=1).reshape((-1, 1)), (1, k)) # probabilistic predictions
    return pp, pt.argmax(axis=1)


def mls(trainx, yt, lambd):
    # Multinomial logistic regression via least squares
    [n , k] = yt.shape
    [_, d] = trainx.shape
    #compute preconditioner a subsample should suffice
    sss = 2 * d #sub-sample size
    rp = np.random.permutation (n)[:sss]
    smallx = trainx[rp, :]
    # why np.sqrt(sss)???
    # $Chol 1/2 n/sss (smallx' smallx)+ lambda sqrt(sss)I
    #C_low = sp.linalg.cho_factor(0.5 * n/sss * (np.dot(smallx.T, smallx))
    #    + lambd * np.sqrt(sss) * np.eye(d), lower=True)
    #true preconditioner
    C_low =sp.linalg.cho_factor(0.5 *           (np.dot(trainx.T, trainx))
        + lambd * np.sqrt(n) * np.eye(d), lower=True)
    C =np.linalg.cholesky(0.5 *           (np.dot(trainx.T, trainx))
        + lambd * np.sqrt(n) * np.eye(d))


    #initialize accelerated gradient variables
    u = np.zeros((d, k))
    w = u.copy()
    #do not mess with these
    li = 1
    linext = 1
    for i in range(100):
        pp,_ = multinom(trainx, u) # get only first elem of tuple
        g = np.dot(trainx.T, (yt - pp))        # gradient d x k
        normg = sp.linalg.norm(g, 'fro')/np.sqrt(d * k * n)
        if normg < 0.01:
            break

        #accelerated gradient updates
        wold = w
        #w = u + sp.linalg.cho_solve((C_low[0].T, not(C_low[1])), (sp.linalg.cho_solve(C_low, g)))
        w = u + np.linalg.solve(C.T, np.linalg.solve(C, g))
                
        gi = (1-li)/linext
        u = (1 - gi) * w + gi * wold
        li = linext
        linext = (1 + np.sqrt(1 + 4 * li**2))/2
    return w


X,y=test_data(1000)
w=mls(X,y,1)
print w
z, ind_z=multinom(X,w)
plt.scatter(X[:,0],X[:,1],c=ind_z)

#yt is n x k where class is from 0:k

#def stagewisemls(trainx,yt,lambd,testx,batchsize,iterations,scale):
#    n, k = yt.shape
#    m, p = testx.shape
#
#    pt = np.zeros((n,k))
#    ps = zeros((m,k))
#    t = 0.5*np.ones((n,1))
#    trainy = yt.argmax(axis=1)+1
#    for batch in range(iterations):
#        r=np.random.randn((p,batchsize))
#        b=2.0*np.pi*np.rand((1,batchsize))
#        trainxb=cos(bsxfun(@plus,scale*trainx*r,b))
#        testxb=cos(bsxfun(@plus,scale*testx*r,b))
#        w=mlsinner(trainxb,yt,2,pt,t)
#        ps=ps + np.dot(testxb, w)
#        pt=pt + np.dot(trainxb, w)
#        zt=pt.max(axis=1)
#        yhatt=pt.argmax(axis=1)
#        errors=(yhatt==trainy).sum()/m
#        printf('iteration %2d,train accuracy: %g\n',batch,1-errors)
#        qt=exp(bsxfun(@minus,pt,zt'))
#        pp=qt./tile(qt.sum(axis=1),(1,k))
#        t=0.1*0.5+0.9*2*(max(pp.T*(1-pp.T))).T
#return ps