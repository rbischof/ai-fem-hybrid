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
            'max_depth': (3, 50),
            'min_child_weight': (0, 10),
            'n_estimators': (30, 300),
            'num_leaves': (10, 100),
            'min_child_samples': (10, 30)
        }


    def train(self, X_train:np.array, y_train:np.array, 
                X_val:np.array, y_val:np.array,
                bayesian_optimization:bool, params:list=None) -> float:

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
                BO.maximize(n_iter=20, init_points=30, acq='ei')
                self.inner_train(**BO.max['params'])
                self.parameters.append(BO.max['params'])
            else:
                self.inner_train(**params[i])
                self.parameters = params


    def inner_train(self, learning_rate, max_depth, min_child_weight, n_estimators, num_leaves, min_child_samples) -> float:          
        model = lgbm.LGBMRegressor(
            learning_rate=learning_rate, 
            max_depth=int(max_depth), 
            min_child_weight=min_child_weight, 
            n_estimators=int(n_estimators),
            num_leaves=int(num_leaves),
            min_child_samples=int(min_child_samples),
            subsample=.5,
            subsample_freq=3,
            force_row_wise=True,
            seed=42,
            verbosity=0
        )

        if len(self.model) <= self.i:
            self.model.append(model)

        self.model[self.i].fit(self.X_train, self.y_train[:, self.i],
                                eval_set=[(self.X_val, self.y_val[:, self.i])], 
                                eval_metric="rmse",
                                callbacks=[lgbm.early_stopping(5)],
                                verbose=False)
                                
        return -np.mean((self.model[self.i].predict(self.X_val) - self.y_val[:, self.i])**2)


    def save_model(self, path:str) -> None:
        models_path = os.path.join(path, 'models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        for i, m in enumerate(self.model):
            if isinstance(m, lgbm.LGBMRegressor):
                m.booster_.save_model(os.path.join(models_path, 'model_'+str(i)+'.txt'))
            else:
                m.save_model(os.path.join(models_path, 'model_'+str(i)+'.txt'), num_iteration=m.best_iteration)
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

        f_importances = []
        for i, ov in enumerate(out_var_names):
            if isinstance(self.model[i], lgbm.LGBMRegressor):
                fi = self.model[i].feature_importances_
            else:
                fi = self.model[i].feature_importance()
            f_importances.append(np.array(fi).reshape(-1, 1))
            plt.figure(figsize=(8, 10)) 
            plt.barh(np.arange(len(fi)), fi)
            plt.yticks(np.arange(len(fi)), in_var_names)
            plt.title(ov)
            plt.xlabel('Importance')
            plt.grid(True, which='major', color='#666666', linestyle='-')
            if path is not None:
                plt.savefig(os.path.join(figures_path, 'feature_importance_'+ov), bbox_inches='tight', dpi=400)
            plt.show()

        plt.figure(figsize=(8, 36)) 
        plt.barh(np.arange(len(f_importances)), np.array(f_importances).mean(axis=-1))
        plt.yticks(np.arange(len(f_importances)), in_var_names)
        plt.title('Mean Feature Importances')
        plt.xlabel('Importance')
        plt.grid(True, which='major', color='#666666', linestyle='-')
        if path is not None:
            plt.savefig(os.path.join(figures_path, 'feature_importance_mean'), bbox_inches='tight', dpi=400)
        plt.show()

        