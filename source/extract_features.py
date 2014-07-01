import numpy as np

def extract_features(X, dictionary, rfSize, CIFAR_DIM, 
                     M,P, encoder='thresh', encParam):
    n_bases = dictionary.shape[0]
    
    # compute features for all training images
    XC = np.zeros((X.shape[0], n_bases*2*4))
    for i in range(n_images):
        if (i% 1000 == 0): 
            print('Extracting features: {0} / {0}\n'.format(i, n_images)
        
        # extract overlapping sub-patches into rows of 'patches'
        patches = np.vstack(( 
            im2col(reshape(X(i,1:1024),CIFAR_DIM(1:2)), (rfSize, rfSize)), 
            im2col(reshape(X(i,1025:2048),CIFAR_DIM(1:2)), (rfSize, rfSize)),
            im2col(reshape(X(i,2049:end),CIFAR_DIM(1:2)), (rfSize, rfSize))) .T

        # do preprocessing for each patch
        
        # normalize for contrast
        patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
        # whiten
        patches = bsxfun(@minus, patches, M) * P;
    
        # compute activation
        
        alpha=encParam;
        z = np.dot(patches, dictionary.T)
        patches = np.hstack(( np.max(z - alpha, 0), -np.max(-z - alpha, 0) ]))
        del z
#        case 'sc'
#        lambda=encParam;
#        z = sparse_codes(patches, D, lambda);
#        patches = [ max(z, 0), -min(z, 0) ];
#        otherwise
#        error('Unknown encoder type.');
#        end
        # patches is now the data matrix of activations for each patch
        
        # reshape to 2*numBases-channel image
        prows = CIFAR_DIM[1]-rfSize+1;
        pcols = CIFAR_DIM[2]-rfSize+1;
        patches = patches.reshape(( prows, pcols, n_bases*2))
        
        # pool over quadrants
        halfr = np.round(prows/2.0)
        halfc = round(pcols/2.0)
        q1 = patches(1:halfr, 1:halfc, :).sum(axis=0).sum(axis=1)
        q2 = patches(halfr+1:end, 1:halfc, :).sum(axis=0).sum(axis=1)
        q3 = patches(1:halfr, halfc+1:end, :).sum(axis=0).sum(axis=1)
        q4 = patches(halfr+1:end, halfc+1:end, :).sum(axis=0).sum(axis=1)
        
        # concatenate into feature vector
        XC[i,:] = np.hstack((q1.ravel(),q2.ravel(),q3.ravel(),q4.ravel()))
    

    return XC 