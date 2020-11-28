import os
import sys
sys.path.append('/opt/ml/model')
import neptune
import pandas as pd
from cv import get_indices
from load_data import load
from model_selection import training
from NN import simple_torchpl
from pl_framework import nn_training


os.environ['NEPTUNE_API_TOKEN']="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzI0ZTMzMDQtZTFmZi00ZjgxLWI4NGQtZGJiOWQyNDJiYjE5In0="
os.environ['NEPTUNE_PROJECT']="iliaavilov/SIBUR"
neptune.init('iliaavilov/SIBUR')

data_path = '/opt/ml/input/data/train/'

random_state = 54321
train_features, train_targets, _ = load(data_path)
cv = get_indices(train_targets, [(pd.to_datetime('2020-03-01 00:00:00'), pd.to_datetime('2020-03-15 00:00:00')),
                                 (pd.to_datetime('2020-03-15 00:00:00'), pd.to_datetime('2020-03-31 00:00:00')),
                                 (pd.to_datetime('2020-04-15 00:00:00'), pd.to_datetime('2020-04-30 00:00:00'))
                                ]
                )
train_targets = train_targets.drop('timestamp', axis = 'columns')
train_features = train_features.drop('timestamp', axis = 'columns')

my_training = training(name = 'NN', nn_model = simple_torchpl, training_nn = nn_training,
                       description = 'Ffill. 1 train set for all folds. LSTM, 2 linear layers. \
                       2 drops. Loss = MAPE. No normalisation. Only first 2 fold and test set(last fold).Dropped data before 2020-02-15 00:00:00',
                       upload_source_files = ['cv.py', 
                                              'load_data.py', 
                                              'model_selection.py',
                                              'NN.py',
                                              'pl_framework.py'])
my_training.set_up_studying(random_state = random_state)

model = 'torch'
def params_func(trial, X):
    return(
        {'n_in': 10,
         'seq_len': trial.suggest_int('seq_len', 1, 70),
         'n_h_1': trial.suggest_int('n_h_1', 2, 1024),
         'n_h_2': trial.suggest_int('n_h_2', 2, 2048),
         'batch_size': trial.suggest_int('batch_size', 10, 800),
         'p_1': trial.suggest_uniform('p_1', 0, 1),
         'p_2':trial.suggest_uniform('p_2', 0, 1),
         'n_out': 4,
         'activation1': trial.suggest_categorical('activation1', ['Tanh','Hardtanh','Hardshrink', 'ELU' , 
                                                                  'SELU', 'ReLU', 'Tanhshrink', 'CELU']),
         'activation2': trial.suggest_categorical('activation2', ['Tanh','Hardtanh','Hardshrink', 'ELU' , 
                                                                  'SELU', 'ReLU', 'Tanhshrink', 'CELU']),
         'optimizer': trial.suggest_categorical('optimizer', ['AdamW','Adadelta','Adagrad',
                                                               'Adam','Adamax',
                                                              'ASGD','LBFGS',
                                                              'RMSprop','Rprop',
                                                              'SGD']),
         'lr': trial.suggest_loguniform('lr', 0.0001, 0.2),
         'weight_decay': trial.suggest_uniform('weight_decay', 0.0001, 1)}
    )
n_trials = 150
my_training.train(X = train_features, 
                  y = train_targets, 
                  cv = cv, 
                  model=model, 
                  params_func = params_func, 
                  n_trials = n_trials)
neptune.stop()

