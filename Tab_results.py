#%% Libreries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import createANN

import joblib
#%% 
# models_name = ['ETWSVM_krein']
# folders_results = ['./results_RFF-TWSVM/']

models_name = ['Ann_v1','Ann_v2']
folders_results = ['./results/','./results_ANN_2layers/']

datasets = os.listdir('./Datasets_')
datasets.sort()
datasets = [dataset.split('.')[0] for dataset in datasets if '.data' in dataset]
datasets.remove('Digit-MultiF2')
datasets.remove('covtype')
#paths_file.remove('./Datasets_/Cryotherapy.xlsx')
#%%
# datasets.remove('Cryotherapy')
#

template = '{}results_{}_f{}.joblib'
colummns = []
for folder in tqdm(folders_results):
    mean = []
    std = []
    for dataset in datasets:
        Sen = []
        Spe = []
        Acc = []
        GM = []
        FM = []
        mdict = joblib.load(template.format(folder,dataset,5))
        for f in range(1,6):
          mdict_aux = mdict['Fold_{}'.format(f)]
          Sen.append( mdict_aux['Sen'][-1] )
          Spe.append( mdict_aux['Spe'][-1] )
          GM.append(mdict_aux['GM'][-1])
          Acc.append( mdict_aux['Acc'][-1] )
          
          FM.append(mdict_aux['F1'][-1])
        mean.append(np.mean(Sen))
        std.append(np.std(Sen))
        mean.append(np.mean(Spe))
        std.append(np.std(Spe))
        mean.append(np.mean(GM))
        std.append(np.std(GM))
        mean.append(np.mean(Acc))
        std.append(np.std(Acc))
        
        mean.append(np.mean(FM))
        std.append(np.std(FM))
    colummns += [mean,std]

# %%
name_columns = []
for model in models_name:
    name_columns.append( model )
    name_columns.append( model )

indices = []
for dataset in datasets:
    indices.append( (dataset,'Sen'))
    indices.append( (dataset,'Spe'))
    indices.append( (dataset,'GM'))
    indices.append( (dataset,'Acc'))
    
    indices.append( (dataset,'FM'))
index = pd.MultiIndex.from_tuples(indices,names=['dataset','metric'])
X = np.vstack(colummns).T

df = pd.DataFrame(X,columns=name_columns,index=index)
print(df)

# %%
df.to_excel('./ANNs.xlsx')
# %%
