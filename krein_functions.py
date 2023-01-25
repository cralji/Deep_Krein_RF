#%% Load base package

import numpy as np
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.linalg import qr

from tensorflow import math,TensorShape,convert_to_tensor
from tensorflow import random,abs,unravel_index,squeeze,matmul
from tensorflow import unique,linspace,constant,cast,zeros_like,where
from tensorflow import reshape,gather_nd,scatter_nd,map_fn,cos
from tensorflow import tensor_scatter_nd_update,pow,exp,Variable
from tensorflow import maximum,sqrt,stack,reduce_sum,reduce_mean,concat,pad
from tensorflow import newaxis,gather,shape,convert_to_tensor
from tensorflow import float32,float64,reduce_max,clip_by_value,transpose
from tensorflow.keras.initializers import Constant as initializer_constant

import tensorflow_probability as tfp

from tensorflow_probability import math as tfmath

#%% Class to based-GORF layer
class KreinMapping(Layer):
    """
    GROF-based Layer for Krein Random features mapping
    """
    def __init__(self,
                dim_out,
                kernel_to_initializer = 'delta_gaussian',
                ww_max = 100,
                num_points = 10000,
                **kwargs) -> None:
        self.dim_out = dim_out
        self.kernel_to_initializer = kernel_to_initializer
        self.ww_max = ww_max
        self.num_points = num_points
        kwargs_kernel = {key:value for key, value in kwargs.items()\
                 if kernel_to_initializer in key}
        self.__dict__.update(kwargs_kernel)
        # extract parameters from kwargs that not used in Tensorflow's\
        # SuperClass
        lits_keys_to_drop = list(kwargs_kernel.keys())
        kwargs = {key:item for key,item in kwargs.items() \
             if key not in lits_keys_to_drop}
        # Pass kwargs to SuperClass Layer.
        super(KreinMapping,self).__init__(**kwargs)
    
    def _M_ortho(self,input_shape):
        """
        Compute M_ortho, from a QR descomposition
        PARAMETER:
            input_shape: int, input dimension of the layer.
        RETURN:
            M_orth: Tensor with shape (input_shape,self.dim_out)
        """
        m2 = maximum(2*input_shape,2*self.dim_out)
        M = RandomNormal(mean=0,stddev = 1)(shape=(m2,m2))
        M_orth,_ = qr(M)
        M_orth = M_orth[:input_shape,:self.dim_out*2]
        return cast(M_orth,self.dtype)
    
    def compute_normal_probaility(self,x,mean,std):
        constant = convert_to_tensor(1/(math.sqrt(2*np.pi)*std),dtype = x.dtype)
        return constant*math.exp(-0.5*(x-mean)*(x-mean)/(std*std))

    def _Compute_delta_gausian_distribution(
        self,
        ww,
        a,
        s,
        input_shape):
        """
        Compute delta_gausian_distribution
        PARAMETER:
            ww: Tensor with shape (self.dim_out,), ||w||
            a: Tensor
            s: Tensor
        RETURN:
            p(||w||): tensor with same shape to ww.
        """
        ww = squeeze(ww) # One-dimensional tensor. 
        a =  squeeze(a) # One-dimensional tensor. 
        s =  squeeze(s) # One-dimensional tensor.
        dtype = ww.dtype
        const = sqrt(2*3.1415926535897)
        func_n = lambda x: x[0]*const*pow(x[1],input_shape-1)*exp(-0.5*ww*ww*x[1]**2)
        result = map_fn(
            func_n,
            elems = stack([a,s],axis=1))
        
        return reduce_sum(result,axis=0)
    
    def _compute_int(self,x,y):
        return tfmath.trapz(abs(reshape(y,(-1,))),x)

    def tf_interp(self,x, xs, ys):
        """
        implemented by: 
        https://brentspell.com/2022/tensorflow-interp/
        """
        # determine the output data type
        # ys = tf.convert_to_tensor(ys)
        dtype = self.dtype
        # normalize data types
        ys = cast(ys, dtype)
        xs = cast(xs, dtype)
        x = cast(x, dtype)

        # pad control points for extrapolation
        xs = concat([[xs.dtype.min], xs, [xs.dtype.max]], axis=0)
        ys = concat([ys[:1], ys, ys[-1:]], axis=0)

        # compute slopes, pad at the edges to flatten
        ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        ms = pad(ms[:-1], [(1, 1)])

        # solve for intercepts
        bs = ys - ms*xs

        # search for the line parameters at each input data point
        # create a grid of the inputs and piece breakpoints for thresholding
        # rely on argmax stopping on the first true when there are duplicates,
        # which gives us an index into the parameter vectors
        i = math.argmax(xs[..., newaxis, :] > x[..., newaxis], axis=-1)
        m = gather(ms, i, axis=-1)
        b = gather(bs, i, axis=-1)

        # apply the linear mapping at each input data point
        y = m*x + b
        return cast(reshape(y, shape(x)), dtype)

    def _sample_weights(self,cdf,ww):
        """
        Compute sample weights
        PARAMETER:
            cdf: Tensor with shape (N,)
            ww: Tensor with shape (N,)
            s: Tensor
        RETURN:
            sample_weights: Tensor with shape (self.dim_out,)
        """
        cdf = squeeze(cdf) # One-dimensional tensor
        w_un = random.uniform((self.dim_out, ), 0, 1)
        ww_sampled = self.tf_interp(w_un,cdf,ww)
        ww_sampled = ww_sampled if self.dtype==ww_sampled.dtype\
             else cast(ww_sampled,self.dtype)
        return ww_sampled
    
    def _compute_initialize_GROF(self,kernel_parameters,D):
        """
        Compute the Grover-of-Fisher matrix
        PARAMETER:
            kernel_parameters: dict with kernel_to_initializer's options
                                all keys be like: 'kermel_name__parameter_name'
        RETURN:
            ww_pos: Tensor with shape (self.dim_out, ), 
                    Positive normed weights ||w_pos||
            ww_neg: Tensor with shape (self.dim_out, ), 
                    Negative normed weights ||w_neg||
        """
        dtype = self.dtype
        ww = linspace(0, self.ww_max, self.num_points)
        ww = cast(ww,self.dtype)


        if self.kernel_to_initializer == 'delta_gaussian':
            s = kernel_parameters[f'{self.kernel_to_initializer}__s']
            if type(s) is list:
                s = convert_to_tensor(s,dtype)
            a = kernel_parameters[f'{self.kernel_to_initializer}__a']
            if type(a) is list:
                a = convert_to_tensor(a,dtype)
            max_v = float32.max if self.dtype == float32\
                 else float64.max
            cons = exp(
                (math.log(max_v) - 0.5*math.log(2*3.1416) 
                - math.log(abs(a)))/(D-1) 
            )
            s = clip_by_value(s,0,reduce_max(cons))

            kernelpower = self._Compute_delta_gausian_distribution(
                ww,
                a = a,
                s = s,
                input_shape = D
            )
        else:
            raise Exception(f"{self.kernel_to_initializer} kernel no implemented")

        kernelpower_pos = zeros_like(kernelpower,dtype = dtype)
        kernelpower_neg = zeros_like(kernelpower,dtype = dtype)
        ind_pos = where(kernelpower>0)
        ind_neg = where(kernelpower<0)
        kernelpower_pos = tensor_scatter_nd_update(
            kernelpower_pos,
            ind_pos,
            gather_nd(kernelpower,ind_pos)
            )
        kernelpower_neg = tensor_scatter_nd_update(
            kernelpower_neg,
            ind_neg,
            -gather_nd(kernelpower,ind_neg)
            )
        kernelpower_pos_coeff = self._compute_int(
            ww,
            kernelpower_pos
            )
        kernelpower_neg_coeff = self._compute_int(
            ww,
            kernelpower_neg
            )
        
        kernelpower_pos /= kernelpower_pos_coeff 
        kernelpower_neg /= kernelpower_neg_coeff

        kernelpower_coeff = kernelpower_pos_coeff + \
            kernelpower_neg_coeff

        kernelpower_pos_coeff = kernelpower_pos_coeff/kernelpower_coeff
        kernelpower_neg_coeff = kernelpower_neg_coeff/kernelpower_coeff

        # Compute the cumulative distribution
        pos_cdf = math.cumsum(kernelpower_pos/math.reduce_sum(
            kernelpower_pos))
        neg_cdf = math.cumsum(kernelpower_neg/math.reduce_sum(
            kernelpower_neg))

        pos_cdf,ind_pos_unique = unique(reshape(pos_cdf,(-1,)))
        neg_cdf,ind_neg_unique = unique(reshape(neg_cdf,(-1,)))

        ind_pos_unique = unique(ind_pos_unique)[0]
        ind_neg_unique = unique(ind_neg_unique)[0]

        ind_pos_unique = transpose(
            unravel_index(
                ind_pos_unique,
                dims=(ind_pos_unique.shape[0],1)
                )
                )
        ind_neg_unique = transpose(
            unravel_index(
                ind_neg_unique,
                dims=(ind_neg_unique.shape[0],1)
                )
                )

        ww_pos = gather_nd(reshape(ww,(-1,1)),ind_pos_unique)
        ww_neg = gather_nd(reshape(ww,(-1,1)),ind_neg_unique)
        self.ww_pos_cdf = ww_pos
        self.ww_neg_cdf = ww_neg
        ww_pos = squeeze(
            self._sample_weights(pos_cdf,ww_pos)
            )
        ww_neg = squeeze(
            self._sample_weights(neg_cdf,ww_neg)
            )
        kernelpower_neg_coeff = cast(math.sqrt(kernelpower_neg_coeff),dtype)
        kernelpower_pos_coeff = cast(math.sqrt(kernelpower_pos_coeff),dtype) 
        
        
        self.pos_cdf = pos_cdf
        self.neg_cdf = neg_cdf

        return ww_pos,ww_neg

    
    def build(self,input_shape):
        D = input_shape[-1] # input-dimensional
        
        if self.kernel_to_initializer is None:
            self.ww_pos = self.add_weight("ww_pos",
                                          shape=(int(self.dim_out),),
                                          initializer = RandomNormal(stddev=1.0),
                                          trainable=True,
                                          constraint = 'NonNeg')
            self.ww_neg = self.add_weight("ww_neg",
                                          shape=(int(self.dim_out),),
                                          initializer = RandomNormal(stddev=1.0),
                                          trainable=True,
                                          constraint = 'NonNeg')
        elif self.kernel_to_initializer == 'delta_gaussian':
            kernel_parameters = {key:value for key,value in\
                self.__dict__.items()\
                    if self.kernel_to_initializer in key}
            ww_pos,ww_neg = self._compute_initialize_GROF(
                kernel_parameters,
                D)
            self.ww_pos = self.add_weight("ww_pos",
                                         shape = (self.dim_out,),
                                         initializer = initializer_constant(ww_pos),
                                         dtype = self.dtype,
                                         trainable=True,
                                         constraint = 'NonNeg')
            self.ww_neg = self.add_weight("ww_neg",
                                         shape = (self.dim_out,),
                                         initializer = initializer_constant(ww_neg),
                                         dtype = self.dtype,
                                         trainable=True,
                                         constraint = 'NonNeg')
        else:
            raise ValueError(f'Not implemented yet, kernel_to_initializer: {self.kernel_to_initializer}')
        
        Morth = self._M_ortho(D)
        self.Morth_pos = Morth[:,:self.dim_out]
        self.Morth_neg = Morth[:,self.dim_out:]
        self.norma_const = cast(sqrt(constant(self.dim_out,self.dtype)),self.dtype)
        
    def call(self,inputs):
        kernel1 = math.multiply(self.ww_pos,self.Morth_pos)
        kernel2 = math.multiply(self.ww_neg,self.Morth_neg)
        #%% Compute masses from naormal distribution
        # mean_pos = reduce_mean(self.ww_pos)
        # mean_neg = reduce_mean(self.ww_neg)
        # std_pos = math.reduce_std(self.ww_pos)
        # std_neg = math.reduce_std(self.ww_neg)

        # mass_pos = self.compute_normal_probaility(self.ww_pos,mean_pos,std_pos)
        # mass_neg = self.compute_normal_probaility(self.ww_neg,mean_neg,std_neg)


        # mass_pos = sqrt(tfp.math.trapz(abs(mass_pos),self.ww_pos))
        # mass_neg = sqrt(tfp.math.trapz(abs(mass_neg),self.ww_neg))

        # total_mass = mass_pos + mass_neg
        # mass_pos = mass_pos/total_mass  
        # mass_neg = mass_neg/total_mass

        #%%
        outputs1 = matmul(a=inputs, b=kernel1)
        outputs2 = matmul(a=inputs, b=kernel2)
        return math.add(cos(outputs1),cos(outputs2))/self.norma_const
    
    def compute_output_shape(self, batch_input_shape):
        return TensorShape(batch_input_shape.as_list()[:-1] + [self.out_dim])

    def get_config(self):
        base_config = super().get_config()
        base_config.update(self.__dict__)
        return base_config

