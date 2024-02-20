#%%
# %pip install -U git+https://github.com/UN-GCPDS/python-gcpds.image_segmentation.git
#%% libraries
import numpy as np

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop,Adam


import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from gcpds.image_segmentation.visualizations import plot_contour
from gcpds.image_segmentation.datasets.segmentation import TomatoSeeds

from gcpds.image_segmentation.models.baseline_unet import unet_baseline

from gcpds.image_segmentation.losses import DiceCoefficient
from gcpds.image_segmentation.losses import GeneralizedCrossEntropy

from gcpds.image_segmentation.metrics import DiceCoefficientMetric

import gcpds.image_segmentation.datasets.segmentation as seg_datasets


from models import unet,krein_rff_unet
from losses import dual_focal_loss

#%% connect wandb
wandb.login(key="0420d50a7cfc013a67dcc9835538e85e13387644")

#%% dataset download
dataset = seg_datasets.OxfordIiitPet()
# %%

def load_data(dataset,
              input_shape = None,
              batch_size =  32
              ):
    train,val,test = dataset()

    if input_shape is None:
        input_shape = (128,128)

    def preprocess(img,mask,shape = input_shape):
        img = tf.image.resize(img,shape)
        mask = tf.image.resize(mask,shape)#Ch 1: Seed, Ch 2: No germinate, Ch 3: germinate
        mask = tf.cast(mask>0, tf.float32)
        mask = mask[...,0]
        mask = tf.expand_dims(mask,axis=-1)
        back_ground = tf.reduce_sum(mask, axis =[-1], keepdims=True)
        back_ground = back_ground == 0
        back_ground = tf.cast(back_ground, tf.float32)
        mask = tf.concat([mask,back_ground], axis=-1) #Ch 1: No germinate, Ch 2: germinate, Ch 3: Background
        return img,mask

    train = train.map(lambda x,y,label,id:preprocess(x,y))
    train = train.batch(batch_size)
    train = train.cache()
    val = val.map(lambda x,y,label,id:preprocess(x,y))
    val = val.batch(batch_size)
    val = val.cache()

    test = test.map(lambda x,y,label,id:preprocess(x,y))
    test_imgs = np.array([element[0] for element in test.as_numpy_iterator()])
    test_masks = np.array([element[1] for element in test.as_numpy_iterator()])
    test = (test_imgs,test_masks)
    return train,val,test

def compute_dice_metric(M_true,M_est,targets):
    mdict = {str(int(target)):-1*DiceCoefficientMetric(target_class=int(target))(M_true,M_est)\
             for target in targets}
    mdict['global'] = -1*DiceCoefficientMetric()(M_true,M_est)
    
    return mdict

#%% sweep
sweep_config = {
  'method': 'bayes', 
  'metric': {
      'name': 'val_dice',
      'goal': 'minimize'
  },
  'early_terminate':{
      'type': 'hyperband',
      'min_iter': 10
  },
  'parameters': {
      "learning_rate": {"min": 0.00001, "max": 0.1},
      "phi_units":{'values':[1,2,3,4]},
      "gamma":{'distribution':'uniform',
               'min':1,
               'max':10},
      "alpha":{'distribution':'uniform',
               'min':0,
               'max':1},
      "rho":{'distribution':'uniform',
               'min':1,
               'max':10},
      "betha":{'distribution':'uniform',
               'min':1,
               'max':10},

  }
}

def sweep_train(config_defaults=None):
    # Set default values
    config_defaults = {
        "phi_units": 1,
        "learning_rate": 0.001,
        "gamma":1,
        "alpha":1,
        "rho":1,
        "betha":1,
        
    }
    # Initialize wandb with a sample project name
    wandb.init(config=config_defaults)  # this gets over-written in the Sweep

    # Specify the other hyperparameters to the configuration, if any
    # wandb.config.epochs = 100
    # wandb.config.log_step = 20
    # wandb.config.val_log_step = 50
    # wandb.config.architecture_name = "MLP"
    # wandb.config.dataset_name = "MNIST"

    input_shape = (128,128)

    dataset = seg_datasets.OxfordIiitPet()
    train,val,test = load_data(dataset,
                               input_shape
                               )
    # train,val,test = dataset()

    # initialize model

    model = krein_rff_unet((input_shape[0],input_shape[1],3),
                            out_channels=2,
                            phi_units=wandb.config.phi_units
                            )
    
    loss = dual_focal_loss(gamma = wandb.config.gamma,
                           alpha = wandb.config.alpha,
                           rho = wandb.config.rho,
                           betha = wandb.config.betha
                           )
    optimizer = RMSprop(learning_rate = wandb.config.learning_rate)
    model.compile(loss = loss,
                    optimizer=optimizer,
                    metrics=[DiceCoefficientMetric(name='dice_cat_dog',target_class=0),
                            DiceCoefficientMetric(name='dice_background',target_class=1),
                            DiceCoefficientMetric(name = 'dice')
                            ]
                    )
    
    wandb_callbacks = [
        WandbMetricsLogger(),
        WandbModelCheckpoint(monitor='val_dice',
                             filepath="my_model_{epoch:02d}",
                             save_best_only=True,
                             mode = 'min'),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.2,
                          patience=5,
                          min_lr=0.001),
                    ]
    callbacks = [EarlyStopping('val_dice_cat_dog',patience = 10)] +\
                wandb_callbacks
    model.fit(train,
                validation_data=val,
                epochs=100,
                callbacks = callbacks,
                verbose=0
                )
    M_est = model.predict(test[0])
    targets =  np.unique(test[1])
    dice_metric =  compute_dice_metric(test[1],M_est,targets=targets)
    wandb.log(dice_metric)
#%%
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow-dog_and_cat")
#%%
wandb.agent(sweep_id, function=sweep_train, count=50)
# %%
