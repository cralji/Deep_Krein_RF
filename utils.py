from krein_functions import Orthogonal,Krein_mapping
from tensorflow import keras
from tensorflow.keras import layers


def create_model_krein(h,
                          scale = None,
                          gamma = 1e-4,
                          l_o = 1e-2,
                          num_classes=2,
                          input_shape = (2,),
                          trainable_scale = True):  
    
    input = keras.Input(input_shape)
    x = layers.Dense(h[0],
                name = 'h1',
                kernel_regularizer = keras.regularizers.l1_l2(gamma,gamma)
                ) (input)

    x = Krein_mapping(out_dim = h[1],
                      scale = scale,
                      kernel_regularizer=Orthogonal(l_o=l_o),
                      trainable_scale=trainable_scale) (x) 

    out = layers.Dense(num_classes,
                      name = 'out',
                      activation = 'softmax',
                      kernel_regularizer =  keras.regularizers.l1_l2(gamma,gamma)
                      ) (x)
    model = keras.Model(inputs = input,outputs =  out)
    return model