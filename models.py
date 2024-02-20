#%% Import libraries
from krein_functions import KreinMapping,Orthogonal,OrthogonalRegularizer
from tensorflow.keras import backend as bk
from tensorflow.keras import layers as la
from tensorflow.keras import models

#%%

def unet(input_shape,
         out_channels=1,
         out_activation = 'softmax',
         activation = 'relu',
         upsamble_mode = 'simple'
         ):
    NET_SCALING = None
    GAUSSIAN_NOISE = 0.1
    EDGE_CROP = 16
    # activation = 'relu'
    # Build U-Net model
    def upsample_conv(filters, kernel_size, strides, padding):
        return la.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    def upsample_simple(filters, kernel_size, strides, padding):
        return la.UpSampling2D(strides)

    if upsamble_mode=='DECONV':
        upsample=upsample_conv
    else:
        upsample=upsample_simple
    input_img = la.Input(input_shape, name = 'RGB_Input')
    pp_in_layer = input_img
    if NET_SCALING is not None:
        pp_in_layer = la.AvgPool2D(NET_SCALING)(pp_in_layer)
        
    # pp_in_layer = la.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = la.BatchNormalization()(pp_in_layer)

    c1 = la.Conv2D(8, (3, 3), activation=activation, padding='same') (pp_in_layer)
    c1 = la.BatchNormalization()(c1)
    c1 = la.Conv2D(8, (3, 3), padding='same') (c1)
    c1 = la.BatchNormalization()(c1)
    p1 = la.MaxPooling2D((2, 2)) (c1)

    c2 = la.Conv2D(16, (3, 3), activation=activation, padding='same') (p1)
    c2 = la.BatchNormalization()(c2)
    c2 = la.Conv2D(16, (3, 3), padding='same') (c2)
    c2 = la.BatchNormalization()(c2)
    p2 = la.MaxPooling2D((2, 2)) (c2)

    c3 = la.Conv2D(32, (3, 3), activation=activation, padding='same') (p2)
    c3 = la.BatchNormalization()(c3)
    c3 = la.Conv2D(32, (3, 3), padding='same') (c3)
    c3 = la.BatchNormalization()(c3)
    p3 = la.MaxPooling2D((2, 2)) (c3)

    c4 = la.Conv2D(64, (3, 3), activation=activation, padding='same') (p3)
    c4 = la.BatchNormalization()(c4)
    c4 = la.Conv2D(64, (3, 3), padding='same') (c4)
    c4 = la.BatchNormalization()(c4)
    p4 = la.MaxPooling2D(pool_size=(2, 2)) (c4)
    c5 = la.Conv2D(128, (3, 3), activation=activation, padding='same') (p4)
    c5 = la.BatchNormalization()(c5)
    c5 = la.Conv2D(128, (3, 3), activation=activation, padding='same') (c5)
    c5 = la.BatchNormalization()(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = la.concatenate([u6, c4])
    c6 = la.Conv2D(64, (3, 3), activation=activation, padding='same') (u6)
    c6 = la.BatchNormalization()(c6)
    c6 = la.Conv2D(64, (3, 3), activation=activation, padding='same') (c6)
    c6 = la.BatchNormalization()(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = la.concatenate([u7, c3])
    c7 = la.Conv2D(32, (3, 3), activation=activation, padding='same') (u7)
    c7 = la.BatchNormalization()(c7)
    c7 = la.Conv2D(32, (3, 3), activation=activation, padding='same') (c7)
    c7 = la.BatchNormalization()(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = la.concatenate([u8, c2])
    c8 = la.Conv2D(16, (3, 3), activation=activation, padding='same') (u8)
    c8 = la.BatchNormalization()(c8)
    c8 = la.Conv2D(16, (3, 3), activation=activation, padding='same') (c8)
    c8 = la.BatchNormalization()(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = la.concatenate([u9, c1], axis=3)
    c9 = la.Conv2D(8, (3, 3), activation=activation, padding='same') (u9)
    c9 = la.BatchNormalization()(c9)
    c9 = la.Conv2D(8, (3, 3), activation=activation, padding='same') (c9)
    c9 = la.BatchNormalization()(c9)

    d = la.Conv2D(out_channels, (1, 1), activation=out_activation) (c9)
    d = la.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = la.ZeroPadding2D((EDGE_CROP, EDGE_CROP),name='output')(d)
    if NET_SCALING is not None:
        d = la.UpSampling2D(NET_SCALING)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    
    return seg_model
        
# %% krein_rff_unet

def krein_rff_unet(input_shape,
         out_channels=1,
         out_activation = 'softmax',
         activation = 'relu',
         upsamble_mode = 'simple',
         phi_units = 64,
         trainable = False,
         trainable_scale = False
         ):
    NET_SCALING = None
    GAUSSIAN_NOISE = 0.1
    EDGE_CROP = 16
    # activation = 'relu'
    # Build U-Net model
    def upsample_conv(filters, kernel_size, strides, padding):
        return la.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    def upsample_simple(filters, kernel_size, strides, padding):
        return la.UpSampling2D(strides)

    if upsamble_mode=='DECONV':
        upsample=upsample_conv
    else:
        upsample=upsample_simple
    input_img = la.Input(input_shape, name = 'RGB_Input')
    pp_in_layer = input_img
    if NET_SCALING is not None:
        pp_in_layer = la.AvgPool2D(NET_SCALING)(pp_in_layer)
        
    # pp_in_layer = la.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = la.BatchNormalization()(pp_in_layer)

    c1 = la.Conv2D(8, (3, 3), activation=activation, padding='same') (pp_in_layer)
    c1 = la.BatchNormalization()(c1)
    c1 = la.Conv2D(8, (3, 3), padding='same') (c1)
    c1 = la.BatchNormalization()(c1)
    p1 = la.MaxPooling2D((2, 2)) (c1)

    c2 = la.Conv2D(16, (3, 3), activation=activation, padding='same') (p1)
    c2 = la.BatchNormalization()(c2)
    c2 = la.Conv2D(16, (3, 3), padding='same') (c2)
    c2 = la.BatchNormalization()(c2)
    p2 = la.MaxPooling2D((2, 2)) (c2)

    c3 = la.Conv2D(32, (3, 3), activation=activation, padding='same') (p2)
    c3 = la.BatchNormalization()(c3)
    c3 = la.Conv2D(32, (3, 3), padding='same') (c3)
    c3 = la.BatchNormalization()(c3)
    p3 = la.MaxPooling2D((2, 2)) (c3)

    c4 = la.Conv2D(64, (3, 3), activation=activation, padding='same') (p3)
    c4 = la.BatchNormalization()(c4)
    c4 = la.Conv2D(64, (3, 3), padding='same') (c4)
    c4 = la.BatchNormalization()(c4)
    p4 = la.MaxPooling2D(pool_size=(2, 2)) (c4)
    # %Bottleneck
    flatten = la.Flatten()(p4)
    rff_krein = KreinMapping(out_dim=int(input_shape[0]/16)*int(input_shape[1]/16)*phi_units,
                             trainable=trainable,
                             trainable_scale = trainable_scale,
                             name = 'Krein_phi')(flatten)
    resha = la.Reshape((int(input_shape[0]/16),int(input_shape[1]/16),-1))(rff_krein)
    # %
    c5 = la.Conv2D(128, (3, 3), activation=activation, padding='same') (resha) #(p4)
    c5 = la.BatchNormalization()(c5)
    c5 = la.Conv2D(128, (3, 3), activation=activation, padding='same') (c5)
    c5 = la.BatchNormalization()(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = la.concatenate([u6, c4])
    c6 = la.Conv2D(64, (3, 3), activation=activation, padding='same') (u6)
    c6 = la.BatchNormalization()(c6)
    c6 = la.Conv2D(64, (3, 3), activation=activation, padding='same') (c6)
    c6 = la.BatchNormalization()(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = la.concatenate([u7, c3])
    c7 = la.Conv2D(32, (3, 3), activation=activation, padding='same') (u7)
    c7 = la.BatchNormalization()(c7)
    c7 = la.Conv2D(32, (3, 3), activation=activation, padding='same') (c7)
    c7 = la.BatchNormalization()(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = la.concatenate([u8, c2])
    c8 = la.Conv2D(16, (3, 3), activation=activation, padding='same') (u8)
    c8 = la.BatchNormalization()(c8)
    c8 = la.Conv2D(16, (3, 3), activation=activation, padding='same') (c8)
    c8 = la.BatchNormalization()(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = la.concatenate([u9, c1], axis=3)
    c9 = la.Conv2D(8, (3, 3), activation=activation, padding='same') (u9)
    c9 = la.BatchNormalization()(c9)
    c9 = la.Conv2D(8, (3, 3), activation=activation, padding='same') (c9)
    c9 = la.BatchNormalization()(c9)

    d = la.Conv2D(out_channels, (1, 1), activation=out_activation) (c9)
    d = la.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = la.ZeroPadding2D((EDGE_CROP, EDGE_CROP),name='output')(d)
    if NET_SCALING is not None:
        d = la.UpSampling2D(NET_SCALING)(d)

    model = models.Model(inputs=[input_img], outputs=[d])
    
    return model
# %%
