import os

import pathlib
import numpy as np

from models.MLModel import MLModel
from models.KNNModel import KNNModel
from models.SVMModel import SVMModel
from models.FCNNModel import FCNNModel
from models.LGBMModel import LGBMModel
from models.XGBoostModel import XGBoostModel


class EnsembleModel(MLModel):
    def __init__(self, base_models:list, ensembler:MLModel, name:str='FCNNModel') -> None:
        super().__init__(name)
        self.parameter_ranges = {
            'depth': (3, 10),
            'width': (32, 512),
            'activation': (0, 2.9),
            'learning_rate': (1e-5, 5e-3),
            'batch_size': (256, 512)
        }
        self.base_models = base_models
        self.ensembler = ensembler


    def train(self, X_train:np.array, y_train:np.array, 
                X_val:np.array, y_val:np.array,
                bayesian_optimization:bool, params:dict=None) -> None:

        self.model = []

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        X_train_hat = X_train
        X_val_hat = X_val
        for m in self.base_models:
            print('creating predictions of base model', m.name)
            X_train_hat = np.hstack([X_train_hat, m.predict(self.X_train)])
            X_val_hat   = np.hstack([X_val_hat, m.predict(self.X_val)])
        
        print('start training of ensembler on data with shape', X_train.shape)
        self.ensembler.train(X_train_hat, y_train, X_val_hat, y_val, bayesian_optimization, params)


    def predict(self, X:np.array) -> np.array:
        """
        Proxy for creating predictions. in case the model consists of various submodels,
        it collects all predictions and concatenates them into a single array.
        """

        preds_base = np.array([m.predict(X) for m in self.base_models]).T.reshape(len(X), -1)
        X = np.hstack([X, preds_base])
        
        return self.ensembler.predict(X)


    def save_model(self, path:str) -> None:
        ensembler_path = os.path.join(path, 'ensembler')
        self.ensembler.save_model(os.path.join(ensembler_path, 'model'))

        base_models_path = os.path.join(ensembler_path, 'base_models')

        for m in self.base_models:
            base_model_path = os.path.join(base_models_path, m.name)
            m.save_model(base_model_path)


    def load_model(self, path:str) -> None:
        ensembler_path = os.path.join(path, 'ensembler')
        self.ensembler.load_model(os.path.join(ensembler_path, 'model'))

        base_models_path  = os.path.join(ensembler_path, 'base_models')
        base_models_paths = list(pathlib.Path(base_models_path))
        self.base_models = []
        for p in base_models_paths:
            base_model = {
                'FCNNModel': FCNNModel(self.X_train, self.y_train),
                'LGBMModel': LGBMModel(self.X_train, self.y_train),
                'XGBoostModel': XGBoostModel(self.X_train, self.y_train),
                'KNNModel': KNNModel(self.X_train, self.y_train),
                'SVMModel': SVMModel(self.X_train, self.y_train)
            }[p.split('/')[-1]]
            self.base_models.append(base_model.load_model(p))

    
    def evaluate(self, path: str, X_test: np.array, y_test: np.array) -> None:
        preds_test = np.array([m.predict(X_test) for m in self.base_models]).T.reshape(len(X_test), -1)
        X_test = np.hstack([X_test, preds_test])
        self.ensembler.evaluate(path, X_test, y_test)
