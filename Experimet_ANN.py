#%% Libraries
from cmath import tanh
import pandas as pd
import numpy as np
import os
from time import time

from krein_functions import Krein_mapping, Orthogonal
from utils import createANN

from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,make_scorer,recall_score,f1_score,balanced_accuracy_score,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,StratifiedShuffleSplit,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import pdist

from scikeras.wrappers import KerasClassifier

import itertools
from tqdm import tqdm

from joblib import dump
import pickle

#%% List Datasets

path_data = './Datasets_'
paths_file = ['{}/{}'.format(path_data,_) for _ in os.listdir(path_data)]
if not('results' in os.listdir()):
    os.mkdir('results')

paths_file.sort()

paths_file.remove('./Datasets_/Digit-MultiF2.data')
paths_file.remove('./Datasets_/covtype.data')
#paths_file.remove('./Datasets_/Cryotherapy.xlsx')
paths_file.remove('./Datasets_/mfeat.info')

#%%
nf = 5
cv1 = StratifiedKFold(n_splits=nf)
cv2 = StratifiedKFold(n_splits=nf)

scoring = {'acc_bal':'balanced_accuracy',
           'f1_w':'f1_weighted'
          } 

#for path_dataset in paths_file:
# path_dataset = paths_file[1]


for path_dataset in paths_file:
    data = pd.read_csv(path_dataset,header = None).to_numpy()
    t = data[:,-1] 
    X = data[:,:-1]
    del data
    # Data
    f=1
    
    labels = np.unique(t)
    #X = X.astype(np.float64)
    #t = t.astype(np.float64)
    nC = labels.size
    # skf = StratifiedKFold(n_splits=10,shuffle=False)
    
    clf = KerasClassifier(model=createANN, #create_model_KreinMapping,
                                        loss = 'sparse_categorical_crossentropy',
                                        metrics = ['accuracy'],
                                        optimizer="SGD",
                                        optimizer__learning_rate = 1e-3,
                                        model__input_shape = X.shape[1:],
                                        num_classes = np.unique(t).size,
                                        model__h = 800,
                                        model__gamma = 1e-4,
                                        fit__epochs = 100,
                                        fit__batch_size = 32,
                                        fit__validation_split = 0.3,
                                        fit__verbose = 0
                                        )
    steps = [
         [('zscore',StandardScaler()),
          ('clf',clf)]
        ]

    name_models = ['ANN_1_layer']
    list_h_simple = [(100,100),
                    (100,300),
                    (100,400),
                    (300,300),
                    (300,400),
                    (400,400),
                    (400,500)]

    h = [100,200,300,400,500]
    

    print('{:*^50}'.format(path_dataset))
    C_list = [0.001,0.1,1,10]
    results_dict = {}
    for train_index, test_index in cv1.split(X, t,t):
        Xtrain,t_train = X[train_index],t[train_index]
        Xtest,t_test = X[test_index],t[test_index]
        
        params_grids = [
                        {'clf__model__h': h,
                        }
                      ]
        #break
        sen = []
        spe = []
        acc = []
        gm = []
        f1 = []
        time_train = []
        T_est = []
        best_params = []
        list_results = []
        for step,params_grid,name_model in zip(steps,params_grids,name_models):
            tik = time()
            pipe = Pipeline(step,memory='pipe_data_pc')
            grid_search = GridSearchCV(pipe,
                                        params_grid,
                                        scoring=scoring,
                                        cv=cv2,
                                        verbose=1,
                                        error_score='raise',
                                        refit ='acc_bal',
                                        n_jobs=1
                                       )
            tok = time()
            time_train.append(tok-tik)
            grid_search.fit(Xtrain,t_train)
            results = grid_search.cv_results_
            t_est = grid_search.best_estimator_.predict(Xtest)
            if nC == 2:
                sen.append( recall_score(t_test,t_est) )
                spe.append( recall_score(t_test,t_est,pos_label=0) )
                acc.append(balanced_accuracy_score(t_test,t_est))
                gm.append(np.sqrt(sen[0]*spe[0]))
                f1.append(f1_score(t_test,t_est,average = 'macro'))
            else:
                sen.append( recall_score(t_test,t_est,average='macro') )
                spe.append( recall_score(t_test,t_est,average='macro') )
                acc.append(balanced_accuracy_score(t_test,t_est))
                gm.append(np.sqrt(sen[-1]*spe[-1]))
                f1.append(f1_score(t_test,t_est,average = 'macro'))
            T_est.append(t_est)
            best_params.append(grid_search.best_params_)
            list_results.append(results)
            print('data_ {} ----> acc: {} \t gm:{} \t f1:{}'.format(path_dataset,
                                                                    acc[-1],
                                                                    gm[-1],
                                                                    f1[-1]))
            model_path = './results/model_{}_{}_f{}.p'.format(name_model,path_dataset[12:-5],f)
            # model_path_tf = './results/model_{}_f{}.h5'.format(path_dataset[7:-4],f)
            # grid_search.best_estimator_[1].model.save(model_path_tf)
            #model_path = './results/model_{}_f{}.p'.format(path_dataset[7:-4],f)  #'model_sujeto_'+str(sbj)+'_cka_featuresCSP_BCI2a_acc.p'
            pickle.dump(grid_search.best_estimator_,open(model_path, 'wb'))
        results_dict['Fold_{}'.format(f)] = {
                                            'best_param':list_results,
                                            'cv_results':results,
                                            'Sen':sen,
                                            'Spe':spe,
                                            'Acc':acc,
                                            'GM':gm,
                                            'F1':f1,
                                            'time':time_train,
                                            'train':(Xtrain,t_train),
                                            'test':(Xtest,t_test),
                                            'T_pred':T_est
                                            }
            
        dump(results_dict,'./results/results_{}_f{}.joblib'.format(path_dataset[12:-5],f))   #'sujeto_'+str(sbj)+'_cka_featuresCSP_BCI2a_acc.joblib')
        f += 1




