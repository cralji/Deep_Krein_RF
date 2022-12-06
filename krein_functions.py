import numpy as np
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.linalg import qr
import tensorflow as tf
import tensorflow_probability as tfp

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
  
  def compute_normal_probaility(self,x,mean,std):
    constant = 1/(tf.math.sqrt(2*np.pi)*std)
    return constant*tf.math.exp(-0.5*(x-mean)*(x-mean)/(std*std))
  
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
                                  trainable = True)
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
    super(Krein_mapping,self).build(input_shape)
  def call(self,inputs):
    kernel1 = (1.0 / self.kernel_scale1) * self.kernel[:,:self.out_dim]
    kernel2 = (1.0 / self.kernel_scale2) * self.kernel[:,self.out_dim:]
    ## Compute masses
    ww = tf.linalg.norm(self.kernel,axis=0)
    ww_pos = tf.sort(ww[:self.out_dim])
    ww_neg = tf.sort(ww[self.out_dim:])
    mean_pos = tf.reduce_mean(ww_pos)
    mean_neg = tf.reduce_mean(ww_neg)
    std_pos = tf.math.reduce_std(ww_pos)
    std_neg = tf.math.reduce_std(ww_neg)

    mass_pos = self.compute_normal_probaility(ww_pos,mean_pos,std_pos)
    mass_neg = self.compute_normal_probaility(ww_neg,mean_neg,std_neg)


    mass_pos = tf.sqrt(tfp.math.trapz(tf.abs(mass_pos),ww_pos))
    mass_neg = tf.sqrt(tfp.math.trapz(tf.abs(mass_neg),ww_neg))

    total_mass = mass_pos + mass_neg
    mass_pos = mass_pos/total_mass  
    mass_neg = mass_neg/total_mass

    ##Compute ouputs
    outputs1 = tf.matmul(a=inputs, b=kernel1)
    outputs2 = tf.matmul(a=inputs, b=kernel2)
    norm_constant = tf.sqrt(tf.cast(tf.shape(kernel1)[1],kernel1.dtype))
    return tf.math.add(mass_pos*tf.cos(outputs1),mass_neg*tf.cos(outputs2))/norm_constant

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
