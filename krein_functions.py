#%%
import numpy as np
from tensorflow.keras.regularizers import Regularizer,OrthogonalRegularizer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.linalg import qr
import tensorflow as tf

class Orthogonal(Regularizer):
  def __init__(self,
               l_o = 1e-2):
    self.l_o = l_o

  def __call__(self,x):
    out_dim = tf.cast(tf.shape(x)[1]/2,tf.int32)
    E1 = tf.linalg.matmul(x[:,:out_dim],x[:,out_dim:],transpose_a=True)-tf.eye(out_dim,dtype = x.dtype)
    ortho1 = (1/tf.cast(out_dim*2,x.dtype))*tf.linalg.trace(tf.linalg.matmul(E1,E1))
    
    return self.l_o*(ortho1)
  
  def get_config(self):
    mdict = {'l_o': self.l_o}
    return mdict

class SumUnit(Constraint):
	def __init__(self,**kwargs):
		super(SumUnit,self).__init__(**kwargs)
	def __call__(self,w):
		den = tf.reduce_sum(w)
		return w/den
	def get_config(self):
		return {}

#%% Krein Layers
class KreinMapping(Layer):

  def __init__(self, 
               out_dim,
               scale = None,
               kernel_regularizer = None,
               trainable_scale = False,
               trainable = False,
               **kwargs):
    super(KreinMapping,self).__init__(**kwargs)
    self.out_dim = out_dim
    self.scale = scale
    self.kernel_regularizer = kernel_regularizer
    self.trainable_scale = trainable_scale
    self.trainable = trainable

  def build(self,input_shape):
    input_dim = input_shape[-1]
    if self.scale is None:
      self.scale1 = np.sqrt(input_dim / 2.0)
      self.scale2 = np.sqrt(input_dim / 2.0)
    elif type(self.scale) is tuple:
      if len(self.scale) == 2:
        self.scale1 = self.scale[0]
        self.scale2 = self.scale[1]
      else:
        self.scale1 = np.sqrt(input_dim / 2.0)
        self.scale2 = np.sqrt(input_dim / 2.0)
    else:
      raise ValueError('scale para')
    if self.kernel_regularizer is None:
      self.kernel_regularizer = OrthogonalRegularizer(factor = 0.01,mode = 'columns')
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         int(self.out_dim*2)],
                                  regularizer = self.kernel_regularizer,
                                  initializer = RandomNormal(stddev=1.0),
                                  trainable = self.trainable)
    self.kernel_scale1 = self.add_weight(
        name='kernel_scale1',
        shape=(1,),
        dtype=tf.float32,
        initializer=tf.constant_initializer(self.scale1),
        trainable = self.trainable_scale,
        constraint='NonNeg')
    self.kernel_scale2 = self.add_weight(
        name='kernel_scale2',
        shape=(1,),
        dtype=tf.float32,
        initializer=tf.constant_initializer(self.scale2),
        trainable = self.trainable_scale,
        constraint='NonNeg')
    # self.masses = self.add_weight(name = 'masses',
    #                               shape = (2,),
    #                               dtype = tf.float32,
    #                               initializer = tf.constant_initializer(0.5),
    #                               trainable = self.trainable_scale,
    #                               constraint = SumUnit())
    super(KreinMapping,self).build(input_shape)
  def call(self,inputs):
    kernel1 = (1.0 / self.kernel_scale1) * self.kernel[:,:self.out_dim]
    kernel2 = (1.0 / self.kernel_scale2) * self.kernel[:,self.out_dim:]
    outputs1 = tf.matmul(a=inputs, b=kernel1)
    outputs2 = tf.matmul(a=inputs, b=kernel2)
    # return tf.math.subtract(self.masses[0]*tf.cos(outputs1),self.masses[1]*tf.cos(outputs2))
    return tf.math.subtract(tf.cos(outputs1),tf.cos(outputs2))
    

  def compute_output_shape(self, batch_input_shape):
    return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.out_dim*2])

  def get_config(self):
    base_config = super().get_config()
    kernel_regularizer = tf.keras.regularizers.serialize(self.kernel_regularizer)
    mdict = {**base_config,
             'out_dim':self.out_dim,
             'scale':self.scale,
             'regularizer':kernel_regularizer,
             'trainable_scale':self.trainable_scale
             }
    base_config.update(mdict)
    return base_config
# %%
