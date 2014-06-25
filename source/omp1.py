import numpy as np
import scipy.sparse

class OMP1:
    def __init__(self, n_clusters=None, iterations=None, batch_size=100):
        self.n_clusters=n_clusters
        self.iterations=iterations
        self.batch_size=batch_size
    
    def  gsvq_step(self, X):
    
        summation = np.zeros(self.clusters_.shape)
        counts = np.zeros((self.n_clusters,1))
        
        for i_start in range(0,X.shape[0],self.batch_size):
            # deal with 
            i_end=min(i_start+self.batch_size, X.shape[0]);
            act_batch_size = i_end-i_start;

            dots = self.transform(X[i_start:i_end,:]).T
            # nclusters by ndata 

            labels = np.abs(dots).argmax(axis=0) # max cluster            

            E = scipy.sparse.csr_matrix((1,
                        np.hstack((labels,np.arange(act_batch_size)))),
                        shape=(act_batch_size,self.n_clusters))
                         
            # labels as indicator matrix
            counts += E.sum(axis=1)

            dots *=  E # dots is now sparse
            summation += np.dot(dots , X[i_start:i_end,:]) 
            # take sum, weighted by dot product
            # nb weighting ensures +/- direction handled properly
        return summation, counts    

    def fit(self, X, y=None):
        ''' finds clusters on unit ball using cosine similarity. 
            rows of X should have l2 norm 1
        ''' 
        self.clusters_=np.randn((self.n_clusters),X.shape[1])
        self.clusters_/=np.sqrt((self.clusters_^2).sum(axis=1)+1e-20)
        
  
        for itr in range(self.iterations):
            print('Running GSVQ:  iteration={0}... \n'.format(itr));
        
            # do assignment + accumulation
            summation,counts = gsvq_step(X)
        
            # reinit empty clusters
            I=np.where((summation^2).sum(axis=1) < 0.001)[0];
            summation[I,:] = np.randn(I.shape[0], X.shape[1]);
        
            # normalize
            self.clusters_ = summation/ np.sqrt((summation^2).sum(axis=1)+1e-20)
        
        return self
    
    def transform(self, X):
        return np.dot(X,self.clusters_.T)

  
    