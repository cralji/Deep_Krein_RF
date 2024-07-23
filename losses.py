#%% Libraries
from tensorflow.keras.losses import Loss,CategoricalCrossentropy
from tensorflow.keras import backend as BK

import tensorflow.linalg as lia
import tensorflow.math as ma
from tensorflow import clip_by_value,reshape
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

class GDL(Loss):
    def __init__(self,
                 epsilon,
                 **kwargs):
        super(GDL,self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def call(self,y_true,y_pred):
        """
        Generalized Dice loss (GDL) [26]
        """
        y_true_f = reshape(y_true, [-1])
        y_pred_f = reshape(y_pred, [-1])
        sum_p = ma.reduce_sum(y_pred_f)
        sum_r = ma.reduce_sum(y_true_f)
        sum_pr = ma.reduce_sum(y_pred_f * y_true_f)
        weights = 1.0 / (ma.square(sum_r) + self.epsilon)

        generalized_dice_loss = 1 - 2 * ( sum_pr * weights) / (sum_r + sum_p + self.epsilon)
        
        return generalized_dice_loss

class overall_loss(Loss):
    def __init__(self,
                ld = 1.25,
                epsilon = 1e-4,
                **kwargs):
        super(overall_loss,self).__init__(**kwargs)
        self.ld = ld
        self.epsilon = epsilon
        self.CE = CategoricalCrossentropy()
        self.GDL = GDL(epsilon=epsilon)
        

    def call(self,y_true,y_pred):
        return self.GDL(y_true,y_pred) + self.ld*self.CE(y_true,y_pred)


class WeightedDiceLoss(Loss):
    def __init__(self, weights=None, name='weighted_dice_loss'):
        super(WeightedDiceLoss, self).__init__(name=name)
        self.weights = tf.constant(weights, dtype=tf.float32) if weights is not None else None

    def call(self, y_true, y_pred):
        # Asegurar que y_pred esté en forma de probabilidad (por ejemplo, usando softmax)
        # y_pred = tf.nn.softmax(y_pred, axis=-1)
        # eps = 1e-6

        # Reducir las dimensiones excepto las clases para calcular la intersección y la unión
        intersection = ma.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        union = ma.reduce_sum(y_true + y_pred, axis=[0, 1, 2])

        # Calcular el coeficiente de Dice por clase
        dice_score = (2. * intersection + BK.epsilon()) / (union + BK.epsilon())

        # Aplicar ponderaciones si están presentes
        if self.weights is not None:
            dice_score = self.weights * dice_score

        # Calcular la pérdida de Dice como 1 menos la media ponderada de los scores de Dice
        return 1 - ma.reduce_mean(dice_score)
        

        
        

# %%