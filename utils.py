from krein_functions import Krein_mapping
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.regularizers import OrthogonalRegularizer


def create_model_KreinMapping(h1,
                          h2,
                          scale1 = None,
                          scale2 = None,
                          gamma = 1e-4,
                          l_o = 1e-2,
                          num_classes=2,
                          input_shape = (2,),
                          trainable_scale = True):  
    
    input = keras.Input(input_shape)
    x = layers.Dense(h1,
                name = 'h1',
                kernel_regularizer = keras.regularizers.l1_l2(gamma,gamma)
                ) (input)
    x = layers.BatchNormalization()(x)
    if scale1 is None or scale2 is None:
      scale = None
    elif scale1==scale2:
      scale = (scale1,scale2+1e-6)
    else:
      scale = (scale1,scale2)
    x = Krein_mapping(out_dim = h2,
                      scale = scale,
                      kernel_regularizer=OrthogonalRegularizer(factor=l_o,mode = 'columns'),
                      trainable_scale=trainable_scale) (x) 

    out = layers.Dense(num_classes,
                      name = 'out',
                      activation = 'softmax',
                      kernel_regularizer =  keras.regularizers.l1_l2(gamma,gamma)
                      ) (x)
    model = keras.Model(inputs = input,outputs =  out)
    return model

def create_model_HybridNN(h,
                          scale = None,
                          gamma = 1e-4,
                          num_classes=2,
                          input_shape = (2,),
                          trainable_scale = True):
    
    input = keras.Input(input_shape)
    x = layers.Dense(h[0],
                            name = 'h1',
                            kernel_regularizer = keras.regularizers.l2(gamma)
                            ) (input)
       
    x = RandomFourierFeatures(output_dim=h[1],
                                    scale=scale,
                                    trainable = trainable_scale,
                                    name = 'h2'
                                    ) (x)
    
    out = layers.Dense(num_classes,
                          name = 'out', 
                          activation = 'softmax',
                          kernel_regularizer =  keras.regularizers.l2(gamma)
                          )(x)
    model = keras.Model(inputs = input,outputs =  out)          
    return model


def createANN1(h1,
              gamma = 1e-4,
              num_classes=2,
              input_shape = (2,)):
    
    input = keras.Input(input_shape)   
    x = layers.Dense(h1,
                name = 'h1',
                kernel_regularizer = keras.regularizers.l1_l2(gamma)
                ) (input)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(num_classes,
                          name = 'out',
                          activation = 'softmax',
                          kernel_regularizer =  keras.regularizers.l1_l2(gamma)
                          )(x)
    model = keras.Model(inputs = input,outputs =  out)          
    return model

def createANN2(h1,
              h2,
              gamma = 1e-4,
              num_classes=2,
              input_shape = (2,)):
    
    input = keras.Input(input_shape)   
    x = layers.Dense(h1,
                        name = 'h1',
                        kernel_regularizer = keras.regularizers.l2(gamma)
                        ) (input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(h2,
              name = 'h2',
              kernel_regularizer = keras.regularizers.l2(gamma)
              ) (x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(num_classes,
                          name = 'out',
                          activation = 'softmax',
                          kernel_regularizer =  keras.regularizers.l2(gamma)
                          )(x)
    model = keras.Model(inputs = input,outputs =  out)          
    return model

#%% random Fourier features Tensorflow implemented
import tensorflow.compat.v2 as tf

import numpy as np
from keras import initializers
from keras.engine import base_layer
from keras.engine import input_spec
from tensorflow.python.util.tf_export import keras_export

_SUPPORTED_RBF_KERNEL_TYPES = ['gaussian', 'laplacian']


@keras_export('keras.layers.experimental.RandomFourierFeatures')
class RandomFourierFeatures(base_layer.Layer):
  r"""Layer that projects its inputs into a random feature space.
  This layer implements a mapping from input space to a space with `output_dim`
  dimensions, which approximates shift-invariant kernels. A kernel function
  `K(x, y)` is shift-invariant if `K(x, y) == k(x - y)` for some function `k`.
  Many popular Radial Basis Functions (RBF), including Gaussian and
  Laplacian kernels, are shift-invariant.
  The implementation of this layer is based on the following paper:
  ["Random Features for Large-Scale Kernel Machines"](
    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
  by Ali Rahimi and Ben Recht.
  The distribution from which the parameters of the random features map (layer)
  are sampled determines which shift-invariant kernel the layer approximates
  (see paper for more details). You can use the distribution of your
  choice. The layer supports out-of-the-box
  approximations of the following two RBF kernels:
  - Gaussian: `K(x, y) == exp(- square(x - y) / (2 * square(scale)))`
  - Laplacian: `K(x, y) = exp(-abs(x - y) / scale))`
  **Note:** Unlike what is described in the paper and unlike what is used in
  the Scikit-Learn implementation, the output of this layer does not apply
  the `sqrt(2 / D)` normalization factor.
  **Usage:** Typically, this layer is used to "kernelize" linear models by
  applying a non-linear transformation (this layer) to the input features and
  then training a linear model on top of the transformed features. Depending on
  the loss function of the linear model, the composition of this layer and the
  linear model results to models that are equivalent (up to approximation) to
  kernel SVMs (for hinge loss), kernel logistic regression (for logistic loss),
  kernel linear regression (for squared loss), etc.
  Examples:
  A kernel multinomial logistic regression model with Gaussian kernel for MNIST:
  ```python
  model = keras.Sequential([
    keras.Input(shape=(784,)),
    RandomFourierFeatures(
        output_dim=4096,
        scale=10.,
        kernel_initializer='gaussian'),
    layers.Dense(units=10, activation='softmax'),
  ])
  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['categorical_accuracy']
  )
  ```
  A quasi-SVM classifier for MNIST:
  ```python
  model = keras.Sequential([
    keras.Input(shape=(784,)),
    RandomFourierFeatures(
        output_dim=4096,
        scale=10.,
        kernel_initializer='gaussian'),
    layers.Dense(units=10),
  ])
  model.compile(
      optimizer='adam',
      loss='hinge',
      metrics=['categorical_accuracy']
  )
  ```
  To use another kernel, just replace the layer creation line with:
  ```python
  random_features_layer = RandomFourierFeatures(
      output_dim=500,
      kernel_initializer=<my_initializer>,
      scale=...,
      ...)
  ```
  Args:
    output_dim: Positive integer, the dimension of the layer's output, i.e., the
      number of random features used to approximate the kernel.
    kernel_initializer: Determines the distribution of the parameters of the
      random features map (and therefore the kernel approximated by the layer).
      It can be either a string identifier or a Keras `Initializer` instance.
      Currently only 'gaussian' and 'laplacian' are supported string
      identifiers (case insensitive). Note that the kernel matrix is not
      trainable.
    scale: For Gaussian and Laplacian kernels, this corresponds to a scaling
      factor of the corresponding kernel approximated by the layer (see concrete
      definitions above). When provided, it should be a positive float. If None,
      a default value is used: if the kernel initializer is set to "gaussian",
      `scale` defaults to `sqrt(input_dim / 2)`, otherwise, it defaults to 1.0.
      Both the approximation error of the kernel and the classification quality
      are sensitive to this parameter. If `trainable` is set to `True`, this
      parameter is learned end-to-end during training and the provided value
      serves as the initial value.
      **Note:** When features from this layer are fed to a linear model,
        by making `scale` trainable, the resulting optimization problem is
        no longer convex (even if the loss function used by the linear model
        is convex).
    trainable: Whether the scaling parameter of the layer should be trainable.
      Defaults to `False`.
    name: String, name to use for this layer.
  """

  def __init__(self,
               output_dim,
               kernel_initializer='gaussian',
               scale=None,
               trainable=False,
               name=None,
               **kwargs):
    if output_dim <= 0:
      raise ValueError(
          f'`output_dim` should be a positive integer. Received: {output_dim}')
    if isinstance(kernel_initializer, str):
      if kernel_initializer.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
        raise ValueError(
            f'Unsupported `kernel_initializer`: {kernel_initializer} '
            f'Expected one of: {_SUPPORTED_RBF_KERNEL_TYPES}')
    if scale is not None and scale <= 0.0:
      raise ValueError('When provided, `scale` should be a positive float. '
                       f'Received: {scale}')
    super(RandomFourierFeatures, self).__init__(
        trainable=trainable, name=name, **kwargs)
    self.output_dim = output_dim
    self.kernel_initializer = kernel_initializer
    self.scale = scale

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    # TODO(pmol): Allow higher dimension inputs. Currently the input is expected
    # to have shape [batch_size, dimension].
    if input_shape.rank != 2:
      raise ValueError(
          'The rank of the input tensor should be 2. '
          f'Received input with rank {input_shape.ndims} instead. '
          f'Full input shape received: {input_shape}')
    if input_shape.dims[1].value is None:
      raise ValueError(
          'The last dimension of the input tensor should be defined. '
          f'Found `None`. Full input shape received: {input_shape}')
    self.input_spec = input_spec.InputSpec(
        ndim=2, axes={1: input_shape.dims[1].value})
    input_dim = input_shape.dims[1].value

    kernel_initializer = _get_random_features_initializer(
        self.kernel_initializer, shape=(input_dim, self.output_dim))

    self.unscaled_kernel = self.add_weight(
        name='unscaled_kernel',
        shape=(input_dim, self.output_dim),
        dtype=tf.float32,
        initializer=kernel_initializer,
        trainable=True)

    # self.bias = self.add_weight(
    #     name='bias',
    #     shape=(self.output_dim,),
    #     dtype=tf.float32,
    #     initializer=initializers.RandomUniform(minval=0.0, maxval=2 * np.pi),
    #     trainable=False)

    if self.scale is None:
      self.scale = _get_default_scale(self.kernel_initializer, input_dim)
    self.kernel_scale = self.add_weight(
        name='kernel_scale',
        shape=(1,),
        dtype=tf.float32,
        initializer=tf.compat.v1.constant_initializer(self.scale),
        trainable=True,
        constraint='NonNeg')
    super(RandomFourierFeatures, self).build(input_shape)

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    inputs = tf.cast(inputs, tf.float32)
    kernel = (1.0 / self.kernel_scale) * self.unscaled_kernel
    outputs = tf.matmul(a=inputs, b=kernel)
    # outputs = tf.nn.bias_add(outputs, self.bias)
    return tf.cos(outputs)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank(2)
    if input_shape.dims[-1].value is None:
      raise ValueError(
          'The last dimension of the input tensor should be defined. '
          f'Found `None`. Full input shape received: {input_shape}')
    return input_shape[:-1].concatenate(self.output_dim)

  def get_config(self):
    kernel_initializer = self.kernel_initializer
    if not isinstance(kernel_initializer, str):
      kernel_initializer = initializers.serialize(kernel_initializer)
    config = {
        'output_dim': self.output_dim,
        'kernel_initializer': kernel_initializer,
        'scale': self.scale,
    }
    base_config = super(RandomFourierFeatures, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _get_random_features_initializer(initializer, shape):
  """Returns Initializer object for random features."""

  def _get_cauchy_samples(loc, scale, shape):
    probs = np.random.uniform(low=0., high=1., size=shape)
    return loc + scale * np.tan(np.pi * (probs - 0.5))

  random_features_initializer = initializer
  if isinstance(initializer, str):
    if initializer.lower() == 'gaussian':
      random_features_initializer = initializers.RandomNormal(stddev=1.0)
    elif initializer.lower() == 'laplacian':
      random_features_initializer = initializers.Constant(
          _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))

    else:
      raise ValueError(
          f'Unsupported `kernel_initializer`: "{initializer}" '
          f'Expected one of: {_SUPPORTED_RBF_KERNEL_TYPES}')
  return random_features_initializer


def _get_default_scale(initializer, input_dim):
  if (isinstance(initializer, str) and
      initializer.lower() == 'gaussian'):
    return np.sqrt(input_dim / 2.0)
  return 1.0

# GCCE_MA_loss
class GCCE_MA_loss(Loss):
  def __init__(self,
               q=0.1,
               K=2,
               **kwargs):
    self.K = K
    self.q = q
    super().__init__(**kwargs)
    
  def call(self, y_true, y_pred):
    # print(y_true,y_pred)
    # q = 0.1
    # pred = y_pred[:, self.R:]
    # pred = tf.clip_by_value(pred, clip_value_min=1e-9, clip_value_max=1)
    # ann_ = y_pred[:, :self.R]
    # # ann_ = tf.clip_by_value(ann_, clip_value_min=1e-9, clip_value_max=1-1e-9)
    # Y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=self.K, axis=1)
    # Y_hat = tf.repeat(tf.expand_dims(pred,-1), self.R, axis = -1)

    p_gcce = y_true*(1 - y_pred**self.q)/self.q
    temp1 = y_pred*p_gcce
    # temp1 = y_pred*tf.math.reduce_sum(p_gcce, axis=1)

    # p_logreg = tf.math.reduce_prod(tf.math.pow(Y_hat, Y_true), axis=1)
    # temp1 = ann_*tf.math.log(p_logreg)  
    # temp2 = (1 - ann_)*tf.math.log(1/K)*tf.reduce_sum(Y_true,axis=1)
    # aux = tf.repeat(tf.reduce_sum(pred*tf.math.log(pred),axis=1,keepdims=True), R, axis = 1)
    # tf.print(tf.shape(aux))
    # print(tf.shape(aux))
    # temp2 = (1 - ann_)*aux*tf.reduce_sum(Y_true,axis=1)
    # temp2 = (tf.ones(tf.shape(ann_)) - ann_)*tf.math.log(1/K)
    # print(tf.reduce_mean(Y_true,axis=1).numpy())
    # Y_true_1 = tf.clip_by_value(Y_true, clip_value_min=1e-9, clip_value_max=1)
    # p_logreg_inv = tf.math.reduce_prod(tf.math.pow(Y_true_1, Y_hat), axis=1)
    # temp2 = (1 - ann_)*tf.math.log(p_logreg_inv) 
    temp2 = (1 - y_pred)*(1-(1/self.K)**self.q)/self.q*y_true
    return tf.math.reduce_sum((temp1 + temp2))

  def get_config(self):
    base_config = super().get_config()
    return {**base_config, 
            "K": self.K,
            "q": self.q}

