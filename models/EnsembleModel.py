import os

import pathlib
import numpy as np

from models.MLModel import MLModel
from models.KNNModel import KNNModel
from models.SVMModel import SVMModel
from models.FCNNModel import FCNNModel
from models.ResNetModel import ResNetModel
from models.MaskNetModel import MaskNetModel
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
        
        print('start training of ensembler on data with shape', X_train_hat.shape)
        self.ensembler.train(X_train_hat, y_train, X_val_hat, y_val, bayesian_optimization, params)


    def predict(self, X:np.array) -> np.array:
        """
        Proxy for creating predictions. in case the model consists of various submodels,
        it collects all predictions and concatenates them into a single array.
        """
        X_hat = X
        for m in self.base_models:
            print('creating predictions of base model', m.name)
            X_hat = np.hstack([X_hat, m.predict(X)])
        
        return self.ensembler.predict(X_hat)


    def save_model(self, path:str) -> None:
        ensembler_path = os.path.join(path, 'ensembler')
        self.ensembler.save_model(os.path.join(ensembler_path, 'model'))

        base_models_path = os.path.join(path, 'base_models')

        for i, m in enumerate(self.base_models):
            base_model_path = os.path.join(base_models_path, str(i), m.__class__.__name__)
            m.save_model(base_model_path)


    def load_model(self, path:str) -> None:
        ensembler_path = os.path.join(path, 'ensembler')
        self.ensembler.load_model(os.path.join(ensembler_path, 'model'))

        base_models_path  = os.path.join(path, 'base_models')
        base_models_paths = list(pathlib.Path(base_models_path).glob('*'))
        self.base_models = []
        for p in base_models_paths:
            base_model_path = list(pathlib.Path(p).glob('*'))[0]
            base_model = {
                'FCNNModel': FCNNModel,
                'ResNetModel': ResNetModel,
                'MaskNetModel': MaskNetModel,
                'LGBMModel': LGBMModel,
                'XGBoostModel': XGBoostModel,
                'KNNModel': KNNModel,
                'SVMModel': SVMModel
            }[str(base_model_path).split('/')[-1]]
            bm = base_model()
            bm.load_model(base_model_path)
            self.base_models.append(bm)


    def evaluate(self, path: str, X_test: np.array, y_test: np.array, in_var_names: list, out_var_names: list) -> None:
        X_hat = X_test
        for m in self.base_models:
            print('creating predictions with base model', m.name)
            X_hat = np.hstack([X_hat, m.predict(X_test)])

        extended_in_var_names = in_var_names.copy()
        for bm in self.base_models:
            extended_in_var_names += [bm.name+'_'+ovn for ovn in out_var_names]
        print('predicting with ensembler')
        self.ensembler.evaluate(path, X_hat, y_test, in_var_names=extended_in_var_names, out_var_names=out_var_names)
