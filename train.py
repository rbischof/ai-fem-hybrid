import os
import argparse
import numpy as np

from models.FCNNModel import FCNNModel
from models.MaskNetModel import MaskNetModel
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments', type=str, help='path where to store the results')

parser.add_argument('--layers', default=12, type=int, help='number of layers')
parser.add_argument('--nodes', default=512, type=int, help='number of nodes')
parser.add_argument('--network', default='fcnn', type=str, help='type of network')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=512, type=int, help='number of sampled points in a batch')

args = parser.parse_args()

X_train, X_val, X_test = np.load('data/X_train.npy', allow_pickle=True), np.load('data/X_val.npy', allow_pickle=True), np.load('data/X_test.npy', allow_pickle=True)
y_train, y_val, y_test = np.load('data/y_train.npy', allow_pickle=True), np.load('data/y_val.npy', allow_pickle=True), np.load('data/y_test.npy', allow_pickle=True)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train = np.hstack([enc.fit_transform(X_train[:, 1:2]), enc.fit_transform(X_train[:, 2:3]), X_train[:, 3:]])
X_val = np.hstack([enc.fit_transform(X_val[:, 1:2]), enc.fit_transform(X_val[:, 2:3]), X_val[:, 3:]])
X_test = np.hstack([enc.fit_transform(X_test[:, 1:2]), enc.fit_transform(X_test[:, 2:3]), X_test[:, 3:]])

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

y_train, y_val, y_test = y_train[:, 1:], y_val[:, 1:], y_test[:, 1:]

def standardize(X:np.array, X_ref:np.array) -> np.array:
    return (X - X_ref.mean(axis=0)) / (X_ref.std(axis=0) + (X_ref.std(axis=0)==0).astype(int))

X_train, X_val, X_test = standardize(X_train, X_train), standardize(X_val, X_train), standardize(X_test, X_train)
y_train, y_val, y_test = standardize(y_train, y_train), standardize(y_val, y_train), standardize(y_test, y_train)

if args.network == 'fcnn':
    net = FCNNModel()
elif args.network == 'masknet':
    net = MaskNetModel()
    
net.train(X_train, y_train, X_val, y_val, bayesian_optimization=False, params={
    'depth': args.layers,
    'width': args.nodes,
    'activation': 0,
    'learning_rate': args.lr,
    'batch_size': args.batch_size,
    'alpha': -3,
    'temperature': -2,
    'rho': -3
})

net.save_model(os.path.join('experiments', args.path))
net.load_model(os.path.join('experiments', args.path))
net.evaluate(os.path.join('experiments', args.path), X_test, y_test)
