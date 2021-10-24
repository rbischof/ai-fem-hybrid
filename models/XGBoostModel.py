import os
import pathlib

import numpy as np
import xgboost as xgb

from models.MLModel import MLModel
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from utils import IN_VAR_NAMES, OUT_VAR_NAMES

class XGBoostModel(MLModel):
    def __init__(self, name:str='XGBoostModel') -> None:
        super().__init__(name)
        self.parameter_ranges = {
            'learning_rate': (1e-5, 0.3),
            'max_depth': (3, 50),
            'n_estimators': (30, 300),
            'num_parallel_tree': (1, 5)
        }


    def train(self, X_train:np.array, y_train:np.array, 
                X_val:np.array, y_val:np.array,
                bayesian_optimization:bool, params:dict=None) -> float:

        self.model = []
        self.parameters = []

        if len(y_train.shape) == 1:
            y_val = y_val.reshape((-1, 1))

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        for i in range(y_val.shape[1]):
            self.i = i

            if bayesian_optimization:
                BO = BayesianOptimization(self.inner_train, self.parameter_ranges)
                BO.maximize(n_iter=30, init_points=10, acq='ei')
                self.inner_train(**BO.max['params'])
                self.parameters.append(BO.max['params'])
            else:
                self.inner_train(**params[i])
                self.parameters = params


    def inner_train(self, learning_rate, max_depth, n_estimators, num_parallel_tree) -> float:          
        model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            num_parallel_tree=int(num_parallel_tree),
            booster='gbtree',
            objective='reg:squarederror',
            eval_metric=['rmse'],
            subsample=0.005,
            feature_selector='greedy',
            seed=42,
            verbosity=1
        )

        if len(self.model) > self.i:
            self.model[self.i] = model
        else:
            self.model.append(model)

        self.model[self.i].fit(self.X_train, self.y_train[:, self.i], 
                                early_stopping_rounds=3, 
                                eval_set=[(self.X_val, self.y_val[:, self.i])], 
                                eval_metric="rmse",
                                verbose=False)
        return -np.mean((self.model[self.i].predict(self.X_val) - self.y_val[:, self.i])**2)

    
    def predict(self, X: np.array) -> np.array:
        X = xgb.DMatrix(X, label=[1]*X.shape[-1])
        return super().predict(X)


    def save_model(self, path:str) -> None:
        models_path = os.path.join(path, 'models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        for i, m in enumerate(self.model):
            m.save_model(os.path.join(models_path, 'model_'+str(i)+'.txt'))
        super().save_model(models_path)


    def load_model(self, path:str) -> None:
        model_paths = list(pathlib.Path(os.path.join(path, 'models')).glob('model_*.txt'))
        self.model = [xgb.Booster() for _ in model_paths]
        for i, m in enumerate(self.model):
            m.load_model(str(model_paths[i]))
        super().load_model(os.path.join(path, 'models'))
            

    def feature_importance(self, path:str, X_test:np.array, in_var_names:list, out_var_names:list) -> None:
        assert X_test.shape[-1] == len(in_var_names)
        assert len(self.model) == len(out_var_names)

        if path is not None:
            figures_path = os.path.join(path, 'figures')
            if not os.path.exists(figures_path):
                os.makedirs(figures_path)

        for i, ov in enumerate(out_var_names):
            fi = self.model[i].get_score(importance_type='gain').values()
            plt.barh(np.arange(len(fi)), fi)
            plt.yticks(np.arange(len(fi)), in_var_names)
            plt.title(ov)
            plt.xlabel('Importance')
            plt.grid(True, which='major', color='#666666', linestyle='-')
            if path is not None:
                plt.savefig(os.path.join(figures_path, 'feature_importance_'+ov), bbox_inches='tight', dpi=400)
            plt.show()