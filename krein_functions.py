import numpy as np
from tensorflow import math
from tensorflow import random,abs,unravel_index
from tensorflow import unique,linspace,constant,cast,zeros_like,where,reshape,gather_nd,scatter_nd,tensor_scatter_nd_update
from tensorflow.keras.initializers import Initializer

from tensorflow_probability import math as tfmath

import tensorflow as tf

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





#%% GRFF
pi = constant(np.pi)
def compute_Normal_pdf(x,mean = 0,std = 1):
  dtype = x.dtype
  mean = constant(mean,dtype=dtype)
  std = constant(std,dtype=dtype)
  pi = constant(np.pi,dtype)
  pdf = ((2*pi*(std**2))**(-0.5))*math.exp(-0.5*(x-mean)*(x-mean)/(std**2))
  return pdf  

def power_delta_gaussian(ww,d,a=None,s = None,mean= 0,dtype=None):
  if dtype is None:
    dtype = ww.dtype
  pi = constant(np.pi,dtype)
  if a is None:
    m = 2
    a = [constant(1,dtype=dtype),constant(-1,dtype=dtype)]
  else:
    m = len(a)
    a = [constant(ai,dtype = dtype) for ai in a]
  if s is None:
    s = [np.sqrt(d/2*np.random.rand())+1e-8 for mi in range(m)]
  if len(a)!=len(s):
    raise ValueError('len(a) is different to len(sigmas)')
  
  sum = 0
  for ai,si in zip(a,s):
    pdf = compute_Normal_pdf(ww,mean = mean,std=si)
    cons = constant( (2*np.pi*(si**2))**(-0.5*(d-1)) ,dtype = dtype)
    sum += ai*cons*pdf
  return sum

def sampler_weights(cdf,ww,d,s,dtype=tf.float64):
  w_un = tf.random.uniform((s,),0,1)
  W_pos = tf.constant(np.interp(w_un,reshape(cdf,(-1,) ),ww),dtype)

  w_pos = random.normal((d,s),0,1,dtype = dtype)
  WW_pos = cast(1/math.sqrt(math.reduce_sum(w_pos**2,axis=0)),dtype)

  WW_pos = tf.repeat(reshape(WW_pos,(1,-1)),d,axis=0)*w_pos

  WW_pos = tf.repeat(reshape(W_pos,(1,-1)),d,axis=0)*WW_pos
  return WW_pos

def GRFF(d,
         s,
         num_points = 10000,
         ww_max = 100,
         sigmas = [1,10],
         a = [1,-1],
         dtype = None):
        #  plot_distributions = False):
  s = int(s/2)

  if dtype is None:
    ww = linspace(0,ww_max,num_points)
    dtype = ww.dtype
    out_dtype = dtype
  else:
    if dtype != tf.float64:
      out_dtype = dtype
      dtype = tf.float64
    else:
      out_dtype = dtype
    ww = linspace(0,ww_max,num_points)
    ww = cast(ww,dtype) 
  pi = constant(np.pi,dtype)
  # Computer kernel power from P(||w||)
  kernelpower = reshape(power_delta_gaussian(ww,d = d,s=sigmas,a = a,dtype = dtype),[-1,1])
  # kernelpower = (2*pi)**(-d/2)*kernelpower
  # Compute kernel positive and negative ( p+(||w||) and p-(||w||) )
  kernelpower_pos = zeros_like(kernelpower,dtype = dtype)
  kernelpower_neg = zeros_like(kernelpower,dtype = dtype)
  ind_pos = where(kernelpower>0)
  ind_neg = where(kernelpower<0)
  kernelpower_pos = tensor_scatter_nd_update(kernelpower_pos,ind_pos,gather_nd(kernelpower,ind_pos) )
  kernelpower_neg = tensor_scatter_nd_update(kernelpower_neg,ind_neg,-gather_nd(kernelpower,ind_neg) )

  compute_int = lambda x,y: tfmath.trapz(abs(reshape(y,(-1,))),x)
  kernelpower_pos_coeff = compute_int(ww,kernelpower_pos)
  kernelpower_neg_coeff = compute_int(ww,kernelpower_neg)

  kernelpower_pos /= kernelpower_pos_coeff 
  kernelpower_neg /= kernelpower_neg_coeff

  kernelpower_coeff = kernelpower_pos_coeff + kernelpower_neg_coeff

  kernelpower_pos_coeff = kernelpower_pos_coeff/kernelpower_coeff
  kernelpower_neg_coeff = kernelpower_neg_coeff/kernelpower_coeff

  # Compute the cumulative distribution
  pos_cdf = math.cumsum(kernelpower_pos/math.reduce_sum(kernelpower_pos))
  neg_cdf = math.cumsum(kernelpower_neg/math.reduce_sum(kernelpower_neg))

  pos_cdf,ind_pos_unique = unique(reshape(pos_cdf,(-1,)))
  neg_cdf,ind_neg_unique = unique(reshape(neg_cdf,(-1,)))

  ind_pos_unique = unique(ind_pos_unique)[0]
  ind_neg_unique = unique(ind_neg_unique)[0]

  ind_pos_unique = tf.transpose(unravel_index(ind_pos_unique,dims=(ind_pos_unique.shape[0],1)))
  ind_neg_unique = tf.transpose(unravel_index(ind_neg_unique,dims=(ind_neg_unique.shape[0],1)))

  ww_pos = gather_nd(reshape(ww,(-1,1)),ind_pos_unique)
  ww_neg = gather_nd(reshape(ww,(-1,1)),ind_neg_unique)

  # Sampler from p_+ and p_- part  
  W_pos = sampler_weights(pos_cdf,ww_pos,d,s,dtype = dtype)
  W_neg = sampler_weights(neg_cdf,ww_neg,d,s,dtype = dtype)
  
#   if plot_distributions:
#     plt.figure(figsize=(9,4))
#     plt.subplot(1,2,1)
#     plt.plot(ww.numpy(),kernelpower_pos.numpy())
#     plt.title('$p_+(||w||)$')
#     plt.subplot(1,2,2)
#     plt.plot(ww.numpy(),kernelpower_neg.numpy())
#     plt.title('$p_-(||w||)$')
#     plt.suptitle('Mass density')
#     plt.show()

#     plt.figure(figsize=(9,4))
#     plt.subplot(1,2,1)
#     plt.plot(ww_pos.numpy(),pos_cdf.numpy())
#     plt.title('$F_+(||w||)$')
#     plt.subplot(1,2,2)
#     plt.plot(ww_neg.numpy(),neg_cdf.numpy())
#     plt.title('$F_-(||w||)$')
#     plt.suptitle('Cumulate distribution')
#     plt.show()
  W = cast(tf.concat([W_pos,W_neg],axis=1),out_dtype)
  kernelpower_pos_coeff = cast(math.sqrt(kernelpower_pos_coeff),out_dtype) 
  kernelpower_neg_coeff = cast(math.sqrt(kernelpower_neg_coeff),out_dtype)
  return W,kernelpower_pos_coeff,kernelpower_neg_coeff



class initializer_GRFF(Initializer):
  def __init__(self,
               num_points = 10000,
               ww_max = 100,
               sigmas = None,
               a = None):
    self.num_points = num_points
    self.ww_max = ww_max
    self.sigmas = sigmas
    self.a = a
  
  def __call__(self,shape,dtype=None):
    W = GRFF(shape[0],
             shape[1],
             num_points = self.num_points,
             ww_max = self.ww_max,
             sigmas = self.sigmas,
             a = self.a,
             dtype = dtype)
    return W
  
  def get_config(self):
    mdict = {'num_points':self.num_points,
             'ww_max':self.ww_max,
             'sigmas':self.sigmas,
             'a':self.a}
    return mdict

#%% Layer Krein
from tensorflow.keras.layers import Layer

class KreinMapping(Layer):

  def __init__(self, 
               out_dim,
               scale = None,
               kernel_regularizer = None,
               factor_reg = 0.01,
               trainable_scale = False,
               trainable = False,
               **kwargs):
    super(KreinMapping,self).__init__(**kwargs)
    self.out_dim = out_dim
    self.scale = scale
    self.kernel_regularizer = kernel_regularizer
    self.factor_reg = factor_reg
    self.trainable_scale = trainable_scale
    self.trainable = trainable

  def build(self,input_shape):
    input_dim = input_shape[-1]
    if self.scale is None:
      self.scale1 = np.sqrt(input_dim / 2.0) + 0.01
      self.scale2 = np.sqrt(input_dim / 2.0) + 0.005
    elif (type(self.scale) is tuple):
      if len(self.scale) == 2:
        self.scale1 = self.scale[0]
        self.scale2 = self.scale[1]
      else:
        self.scale1 = np.sqrt(input_dim / 2.0) + 0.01
        self.scale2 = np.sqrt(input_dim / 2.0) + 0.005
    elif type(self.scale) is list:
      if len(self.scale) == 2:
        self.scale1 = self.scale[0]
        self.scale2 = self.scale[1]
      else:
        self.scale1 = np.sqrt(input_dim / 2.0) + 0.01
        self.scale2 = np.sqrt(input_dim / 2.0) + 0.005
    else:
      raise ValueError('scale para')
    
    if self.kernel_regularizer is None:
      # self.kernel_regularizer = OrthogonalRegularizer(factor = self.factor_reg,
      #                                                 mode = 'columns'
      #                                                 )
      self.kernel_regularizer = Orthogonal(l_o=self.factor_reg)
    
    W,pos,neg = GRFF(int(input_shape[-1]),
                     int(self.out_dim*2),
                     sigmas=[self.scale1,self.scale2],
                     dtype = tf.float32)
    self.kernel = tf.Variable(W,
                              regularizer = self.kernel_regularizer,
                              trainable=self.trainable,
                              dtype=tf.float32
                              )
    self.pos = pos
    self.neg = neg
    # self.kernel = self.add_weight("kernel",
    #                               shape=[int(input_shape[-1]),
    #                                      int(self.out_dim*2)],
    #                               regularizer = self.kernel_regularizer,
    #                               initializer = initializer_GRFF(sigmas = [self.scale1,self.scale2]),
    #                               trainable=True)
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
    super(KreinMapping,self).build(input_shape)
  def call(self,inputs):
    kernel1 = self.kernel[:,:self.out_dim]#*(1.0 / self.kernel_scale1)
    kernel2 = self.kernel[:,self.out_dim:]#*(1.0 / self.kernel_scale2)
    outputs1 = tf.matmul(a=inputs, b=kernel1)
    outputs2 = tf.matmul(a=inputs, b=kernel2)
    return tf.math.subtract(self.pos*tf.cos(outputs1),self.neg*tf.cos(outputs2))

  def compute_output_shape(self, batch_input_shape):
    return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.out_dim*2])

  def get_config(self):
    base_config = super().get_config()
    kernel_regularizer = tf.keras.regularizers.serialize(self.kernel_regularizer)
    mdict = {**base_config,
             'out_dim':self.out_dim,
             'scale':self.scale,
             'regularizer':kernel_regularizer,
             'trainable_scale':self.trainable_scale,
             'trainable' :self.trainable
             }
    base_config.update(mdict)
    return base_config