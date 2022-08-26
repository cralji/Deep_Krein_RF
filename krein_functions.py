import numpy as np
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.linalg import qr
import tensorflow as tf

from tensorflow_probability import math as tfmath
from tensorflow_probability import stats as tfstats

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

def Compute_masses(W,bins = 10):
  norm_W = tf.linalg.norm(W,axis=0)
  edges = tf.linspace(tf.reduce_min(norm_W),tf.reduce_max(norm_W),bins)
  freq = tfstats.histogram(norm_W,edges=edges)
  return tfmath.trapz(freq/tf.reduce_sum(freq),edges[:-1])


# Krein Layers
# @tf.keras.utils.register_keras_serializable(package='Custom',name = 'Krein_mapping')
class Krein_mapping(Layer):

  def __init__(self, 
               out_dim,
               scale = None,
               kernel_regularizer = None,
               trainable_scale = False,
               **kwargs):
    super(Krein_mapping,self).__init__(**kwargs)
    self.out_dim = out_dim
    self.scale = scale
    self.kernel_regularizer = kernel_regularizer
    self.trainable_scale = trainable_scale

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
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         int(self.out_dim*2)],
                                  regularizer = self.kernel_regularizer,
                                  initializer = RandomNormal(stddev=1.0),
                                  trainable=False)
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
    self.masses_p = tf.random.uniform(())
    self.masses_n = tf.random.uniform(())  
    super(Krein_mapping,self).build(input_shape)
  def call(self,inputs):
    W_p = self.kernel[:,:self.out_dim]
    W_n = self.kernel[:,self.out_dim:]
    kernel1 = (1.0 / self.kernel_scale1) * W_p
    kernel2 = (1.0 / self.kernel_scale2) * W_n
    masses_p = Compute_masses(kernel1)
    masses_n = Compute_masses(kernel2)
    tf.print(masses_p,masses_n)
    masses = masses_p + masses_n
    self.masses_p = masses_p/masses
    self.masses_n = masses_n/masses
    outputs1 = tf.matmul(a=inputs, b=kernel1)
    outputs2 = tf.matmul(a=inputs, b=kernel2)
    return tf.math.subtract(self.masses_p*tf.cos(outputs1),self.masses_n*tf.cos(outputs2))

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
