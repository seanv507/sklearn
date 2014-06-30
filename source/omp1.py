import numpy as np
import scipy.sparse

# to do add metrics, and verbose flag...

class OMP1:
    def __init__(self, n_clusters=None, iterations=None, batch_size=100):
        self.n_clusters=n_clusters
        self.iterations=iterations
        self.batch_size=batch_size
    
    def  gsvq_step(self, X):
    
        summation = np.zeros(self.cluster_centers_.shape)
        counts = np.zeros((self.n_clusters,1))
        self.inertia_=0        
        
        for i_start in range(0,X.shape[0],self.batch_size):
            # deal with 
            i_end=min(i_start+self.batch_size, X.shape[0]);
            act_batch_size = i_end-i_start;

            dots = self.transform(X[i_start:i_end,:]).T
            # nclusters by ndata 


            labels = np.abs(dots).argmax(axis=0) # max cluster
            # turn labels into 1-hot encoding

            E = scipy.sparse.csr_matrix((np.ones((act_batch_size,)),
                        np.vstack((labels,np.arange(act_batch_size)))),
                        shape=(self.n_clusters,act_batch_size))
                         
            
            counts += E.sum(axis=1)
            
            dots =  E.multiply(dots) # dots is now sparse
            # inertia = x^2  -2x.y +  y^2, so 
            # 2(1-abs(dots)) (ASSUMING input data (and clusters) have unit norm)
            # 
            self.inertia_-=np.abs(dots).sum()
            summation += np.dot(dots , X[i_start:i_end,:]) 
            # take sum of X vectors assigned to each cluster, weighted by dot product
            # nb weighting ensures +/- direction handled properly
        self.inertia_=2*(X.shape[0]-self.inertia_)
        # inertia will not sum over partial fits
        return summation, counts
    
        
    def partial_fit(self,X, y=None):
        self.inertia_=0
        for itr in range(self.iterations):
            print('Running GSVQ:  iteration={0}...inertia={1}\n'.
                format(itr, self.inertia_))
            
            # do assignment + accumulation
            summation,counts = self.gsvq_step(X)
        
            # reinit empty clusters
            I=np.where((summation**2).sum(axis=1) < 0.001)[0]
            summation[I,:] = np.random.randn(I.shape[0], X.shape[1])
        
            # normalize
            self.cluster_centers_ = summation/ \
                np.sqrt((summation**2).sum(axis=1)[:,np.newaxis]+1e-20)
        
        return self
    
    def fit(self, X, y=None):
        ''' finds clusters on unit ball using cosine similarity. 
            rows of X should have l2 norm 1
        ''' 
        self.cluster_centers_=np.random.randn((self.n_clusters),X.shape[1])
        self.cluster_centers_ /= \
            np.sqrt((self.cluster_centers_**2).sum(axis=1)[:,np.newaxis]+1e-20)
        
        return self.partial_fit(X,y)
  
    
    def transform(self, X):
        return np.dot(X,self.cluster_centers_.T)

  
    