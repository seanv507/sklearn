from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import scipy.stats

def multinormal_pdf(r, mean, cov):
    """ Probability density function for a multidimensional Gaussian distribution."""
    dim  = r.shape[-1]
    dev  = r - mean
    maha = np.einsum('...k,...kl,...l->...', dev, np.linalg.pinv(cov), dev)
    return (2 * np.pi)**(-0.5 * dim) * np.linalg.det(cov)**(0.5) * np.exp(-0.5 * maha)




def gen_likelihood(lim_x,lim_y,mn_A,mn_B,cov_A,cov_B,N_A,N_B):
    p_A=N_A/(N_A+N_B)
    p_B=1-p_A
    
    x, y = np.mgrid[lim_x[0]:lim_x[1]:.01, lim_y[0]:lim_y[1]:.01,]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    p_x_A=multinormal_pdf(pos, mn_A, cov_A)*p_A
    p_x_B=multinormal_pdf(pos, mn_B, cov_B)*p_B
    
    p_A_g_x=p_x_A/(p_x_A+p_x_B)
    # pA g x = p(x& a)/p(x)
    logist=np.log(p_A_g_x/(1-p_A_g_x))
   # levels=np.linspace(-20,20,40)
    #logist=logist.reshape((n_pts,n_pts))    
    plt.contourf(x, y, logist)#,levels=levels)
    
    #x=np.linspace(lim_x[0],lim_x[1],n_pts)
    #y=np.linspace(lim_y[0],lim_y[1],n_pts)
    #xg,yg=np.meshgrid(x,y,'ij')
    
    return logist

def plot_clf(clf,ymin,ymax):
    x1=-(clf.intercept_+clf.coef_[0][1]*ymin)/clf.coef_[0][0]
    x2=-(clf.intercept_+clf.coef_[0][1]*ymax)/clf.coef_[0][0]
    plt.plot([x1, x2],[ymin,ymax])

def gen_data(N):
    mn_A=[0,0]
    mn_B=[2,0]
    N_A=N
    N_B=N
    cov_A=[[1,0],[0,1]]
    cov_B=[[1,0.75],[0.75,1]]
    dat_A=np.random.multivariate_normal(mn_A,cov_A,N_A)
    dat_B=np.random.multivariate_normal(mn_B,cov_B,N_B)
    dat=np.concatenate((dat_A,dat_B))
    tgt=np.hstack((np.ones((N_A,)),np.zeros((N_B,))))
    return dat,tgt

N_A=1000

mn_A=[0,0]
mn_B=[2,0]

cov_A=[[1,0],[0,1]]
cov_B=[[1,0.75],[0.75,1]]

plt.clf()
clfs=[]
dat_A=np.random.multivariate_normal(mn_A,cov_A,N_A)
for i in range(0,3):
    plt.subplot(2,2,i+1)
    N_B=N_A*10**i
    dat_B=np.random.multivariate_normal(mn_B,cov_B,N_B)
    clf=lm.LogisticRegression()
    dat=np.concatenate((dat_A,dat_B))
    tgt=np.hstack((np.ones((N_A,)),np.zeros((N_B,))))
    clf.fit(dat,tgt)
    clfs.append(clf)
    logist=gen_likelihood([-6, 6],[-4,4],mn_A,mn_B,cov_A,cov_B,N_A,N_B)
    plt.plot(dat_A[:,0],dat_A[:,1],'k.')
    plt.plot(dat_B[:,0],dat_B[:,1],'w.')
    plot_clf(clf,-4,4)
     
