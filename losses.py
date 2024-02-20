#%% Libraries
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as BK
import tensorflow.linalg as lia
import tensorflow.math as ma
from tensorflow import clip_by_value
#%% Losses
# Dual Focal Loss Segmentation

class dual_focal_loss(Loss):
    def __init__(self,
                 gamma = 1,
                 alpha = 1,
                 betha = 1,
                 rho = 1,
                 **kwargs
                 ):
        super(dual_focal_loss,self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.betha = betha
        self.rho = rho
    def call(self,
             Y,
             M):
        epsilon = BK.epsilon()
        M = clip_by_value(M, epsilon, 1. - epsilon)
        loss = Y*ma.log(M)\
            + self.betha*(1-Y)*ma.log(self.rho - M)\
            - self.alpha*ma.pow(ma.abs(Y - M),self.gamma)
        loss1 = -1*ma.reduce_sum(loss,axis = -1)
        loss2 = ma.reduce_mean(loss1,axis=[1,2])
        return ma.reduce_mean(loss2)

        
        
