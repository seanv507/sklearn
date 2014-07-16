# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 21:33:38 2014

@author: sean
"""

class ImageTransform:
    def __init__(self,  height=32,width=32, n_channels=3):
        self.height=height
        self.width=width
        self.n_channels=n_channels
        
    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
        return { 'height' : self.height, 
        'width' : self.width,
        'n_channels': self.n_channels}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)

class SIFTTransform( ImageTransform):
    def __init__(self, height=32, width=32, n_channels=3):
        ImageTransform.__init__( height,width, n_channels)
        
    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
        
        return ImageTransform.get_params(self, deep=True)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
            ImageTransform.setattr(parameter, value)
            
    def transform(X):
     
        kp_list=[]
        n_images=X.shape[0]
        for image in X:
            img_rgb=image.reshape((ImageTransform.height,
                                   ImageTransform.width,
                                   ImageTransform.n_channels))
            img_bgr=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)            
            gray= cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
            
            sift = cv2.SIFT()

            kps = sift.detect(gray,None)
            fields={'pt_x':np.float,'pt_y':np.float,
            'angle':np.float, 'response':np.float,
            'octave':np.int, 'class_id':np.int}
            
            pd.DataFrame(zeros((len(kps),6),fields))
                
            for kp in kps:
                x,y =kp.pt
                angle=kp.angle
                response=kp.response
                octave=kp.octave
                class_id=kp.class_id
                
        return X
        
