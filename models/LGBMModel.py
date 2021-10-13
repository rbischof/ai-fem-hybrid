import os
import pathlib
from utils import IN_VAR_NAMES, OUT_VAR_NAMES
import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt

from models.MLModel import MLModel
from bayes_opt import BayesianOptimization

class LGBMModel(MLModel):
    def __init__(self, name:str='LGBMModel') -> None:
        super().__init__(name)
        self.parameter_ranges = {
            'learning_rate': (1e-5, 0.3),
            'gamma': (0, 10),
            'max_depth': (3, 50),
            'min_child_weight': (0, 10),
            'n_estimators': (30, 300),
            'num_leaves': (10, 100),
            'min_data_in_leaf': (10, 30)
        }


    def train(self, X_train:np.array, y_train:np.array, 
                X_val:np.array, y_val:np.array,
                bayesian_optimization:bool, params:dict=None) -> float:

        self.model = []

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
                BO.maximize(n_iter=40, init_points=20, acq='ei')
                self.inner_train(**BO.max['params'])
                self.parameters = BO.max['params']
            else:
                self.inner_train(**params)
                self.parameters = params


    def inner_train(self, learning_rate, gamma, max_depth, min_child_weight, n_estimators, num_leaves, min_data_in_leaf) -> float:          
        model = lgbm.LGBMRegressor(
            learning_rate=learning_rate, 
            gamma=gamma, 
            max_depth=int(max_depth), 
            min_child_weight=min_child_weight, 
            n_estimators=int(n_estimators),
            num_leaves=int(num_leaves),
            min_data_in_leaf=int(min_data_in_leaf),
            bagging_fraction=.5,
            bagging_freq=3,
            seed=42,
            verbosity=0,
            num_threads=8
        )

        if len(self.model) > self.i:
            self.model[self.i] = model
        else:
            self.model.append(model)

        self.model[self.i].fit(self.X_train, self.y_train[:, self.i],
                                early_stopping_rounds=5, 
                                eval_set=[(self.X_val, self.y_val[:, self.i])], 
                                eval_metric="rmse",
                                verbose=False)
        return -np.mean((self.model[self.i].predict(self.X_val) - self.y_val[:, self.i])**2)


    def save_model(self, path:str) -> None:
        models_path = os.path.join(path, 'models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        for i, m in enumerate(self.model):
            m.booster_.save_model(os.path.join(models_path, 'model_'+str(i)+'.txt'))
        super().save_model(models_path)


    def load_model(self, path:str) -> None:
        model_paths = list(pathlib.Path(os.path.join(path, 'models')).glob('model_*.txt'))
        self.model = [lgbm.Booster(model_file=str(p)) for p in model_paths]
        super().load_model(os.path.join(path, 'models'))

    def feature_importance(self, path:str, X_test:np.array, in_var_names:list, out_var_names:list) -> None:
        assert X_test.shape[-1] == len(in_var_names)
        assert len(self.model) == len(out_var_names)

        if path is not None:
            figures_path = os.path.join(path, 'figures')
            if not os.path.exists(figures_path):
                os.makedirs(figures_path)

        for i, ov in enumerate(out_var_names):
            fi = self.model[i].feature_importances_

            plt.barh(np.arange(len(fi)), fi)
            plt.yticks(np.arange(len(fi)), in_var_names)
            plt.title(ov)
            plt.xlabel('Importance')
            plt.grid(True, which='major', color='#666666', linestyle='-')
            if path is not None:
                plt.savefig(os.path.join(figures_path, 'feature_importance_'+ov), bbox_inches='tight', dpi=400)
            plt.show()