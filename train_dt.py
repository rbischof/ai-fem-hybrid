import os
import argparse
import numpy as np

from models.LGBMModel import LGBMModel

IN_VAR_NAMES  = {'0':   ['Material Model Nr. '+str(i) for i in range(6)]+['CMM Usermat Model '+str(i) for i in range(5)]+\
                        ['Normal Strain X', 'Normal Strain Y','Shear Strain XY'],
                 '1':   ['Material Model Nr. '+str(i) for i in range(6)]+['CMM Usermat Model '+str(i) for i in range(5)]+\
                        ['Reinforcement Area', 'Reinforcement Diameter','Effective Reinforcement TCM', 'Reinforcement Angle',
                         'Normal Strain X', 'Normal Strain Y','Shear Strain XY']}

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments', type=str, help='path where to store the results')
parser.add_argument('--reinforcement_layer', default='0', type=str, help='train on dataset without (0) or with (1) reinforcement')
parser.add_argument('--model', default='lgbm', type=str, help='model to use for training')
args = parser.parse_args()

X_train = np.load('data/X_train'+args.reinforcement_layer+'_prepro.npy', allow_pickle=True)[:, 1:]
X_val   = np.load('data/X_val'+args.reinforcement_layer+'_prepro.npy', allow_pickle=True)[:, 1:]
X_test  = np.load('data/X_test'+args.reinforcement_layer+'_prepro.npy', allow_pickle=True)[:, 1:]
y_train = np.load('data/y_train'+args.reinforcement_layer+'_prepro.npy', allow_pickle=True)[:, 1:]
y_val   = np.load('data/y_val'+args.reinforcement_layer+'_prepro.npy', allow_pickle=True)[:, 1:]
y_test  = np.load('data/y_test'+args.reinforcement_layer+'_prepro.npy', allow_pickle=True)[:, 1:]

def standardize(X:np.array, X_ref:np.array) -> np.array:
    return (X - X_ref.mean(axis=0)) / (X_ref.std(axis=0) + (X_ref.std(axis=0)==0).astype(int))

X_val = np.hstack([X_val[:, :11], standardize(X_val[:, 11:], X_train[:, 11:])])
X_test = np.hstack([X_test[:, :11], standardize(X_test[:, 11:], X_train[:, 11:])])
X_train = np.hstack([X_train[:, :11], standardize(X_train[:, 11:], X_train[:, 11:])])
y_train, y_val, y_test = standardize(y_train, y_train), standardize(y_val, y_train), standardize(y_test, y_train)

if args.model == 'lgbm':
    net = LGBMModel()
# elif args.model == 'xgboost':
#     net = XGBoostModel()
    
net.train(X_train, y_train, X_val, y_val, bayesian_optimization=True)
net.save_model(os.path.join('experiments', args.path))
net.evaluate(os.path.join('experiments', args.path), X_test, y_test, in_var_names=IN_VAR_NAMES[args.reinforcement_layer])
