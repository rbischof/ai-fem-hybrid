import os
import argparse
import numpy as np

from models.FCNNModel import FCNNModel
from models.ResNetModel import ResNetModel
from models.MaskNetModel import MaskNetModel
from sklearn.preprocessing import OneHotEncoder

IN_VAR_NAMES  = {'0':   ['Material Model Nr. '+str(i) for i in range(6)]+['CMM Usermat Model '+str(i) for i in range(5)]+\
                        ['Normal Strain X', 'Normal Strain Y','Shear Strain XY'],
                 '1':   ['Material Model Nr. '+str(i) for i in range(6)]+['CMM Usermat Model '+str(i) for i in range(5)]+\
                        ['Reinforcement Area', 'Reinforcement Diameter','Effective Reinforcement TCM', 'Reinforcement Angle',
                         'Normal Strain X', 'Normal Strain Y','Shear Strain XY']}

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments', type=str, help='path where to store the results')
parser.add_argument('--reinforcement_layer', default='0', type=str, help='train on dataset without (0) or with (1) reinforcement')

parser.add_argument('--layers', default=12, type=int, help='number of layers')
parser.add_argument('--nodes', default=512, type=int, help='number of nodes')
parser.add_argument('--network', default='fcnn', type=str, help='type of network')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=256, type=int, help='number of sampled points in a batch')

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

if args.network == 'fcnn':
    net = FCNNModel()
elif args.network == 'masknet':
    net = MaskNetModel()
elif args.network == 'resnet':
    net = ResNetModel()
    
net.train(X_train, y_train, X_val, y_val, bayesian_optimization=False, params={
    'depth': args.layers,
    'width': args.nodes,
    'activation': 0,
    'learning_rate': args.lr,
    'batch_size': args.batch_size,
    'alpha': -2,
    'temperature': -3,
    'rho': -4
})

net.evaluate(os.path.join('experiments', args.path), X_test, y_test, in_var_names=IN_VAR_NAMES[args.reinforcement_layer])
