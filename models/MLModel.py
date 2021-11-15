import os

import shap
import pickle
import pathlib
import numpy as np

from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from utils import append_to_results, IN_VAR_NAMES, OUT_VAR_NAMES
from statistics import accuracy_diagonal_plot, mean_correction_factor_plot, qq_plot, ratio_plot


class MLModel():
    """
    A parent class for representing a Machine Learning Model

    Attributes
    ----------
    parameter_ranges : list
        a list of dictionarys containing each hyperparameter and its value if model has already been trained
    parameter_ranges : dict
        a dictionary containing each hyperparameter and its range
        used for bayesian optimization
    model : list
        list of models. contains 1 model if it supports multivariate output.
        otherwise, the list contains 1 model per output variable.
    """

    def __init__(self, name:str='MLModel', parameters:list=[{}]) -> None:
        """
        Builder function
        """
        self.name = name
        self.parameters = parameters
        self.parameter_ranges = {}
        self.model = []

    def train(self, X_train:np.array, y_train:np.array, 
                X_val:np.array, y_val:np.array,
                bayesian_optimization:bool, params:dict=None) -> float:
        """
        Wrapper function for training the models. Depending on arguments, it trains a single model
        with the provided parameters or performs bayesian optimization.

        Arguments
        ---------
        X_train : np.array
            input for training
        y_train : np.array
            output for training
        X_val : np.array
            input for validation
        y_val : np.array
            output for validation
        bayesian_optimization : bool
            run bayesian optimization or not
        params : dict
            if bayesian optimization is False, this argument must contain the values of
            the required hyperparameters that the model should be trained with.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        if bayesian_optimization:
            BO = BayesianOptimization(self.inner_train, self.parameter_ranges)
            BO.maximize(n_iter=30, init_points=10, acq='ei')
            self.inner_train(**BO.max['params'])
            self.parameters = [BO.max['params']]
        else:
            assert params is not None
            if isinstance(params, dict):
                params = [params]
            self.inner_train(**(params[0]))
            self.parameters = [params]


    def inner_train(self) -> float:
        """
        Inner training function that gets called by self.train with all the required hyperparameters
        for building and training the model
        """
        pass

    def predict(self, X:np.array) -> np.array:
        """
        Proxy for creating predictions. in case the model consists of various submodels,
        it collects all predictions and concatenates them into a single array.
        """
        pred = np.array([self.model[i].predict(X) for i in range(len(self.model))])
        
        if len(self.model) > 1:
            return pred.T
        else:
            return pred[0]


    def save_model(self, path:str) -> None:
        """
        Stores the model at given path.
        """
        for i in range(len(self.parameters)):
            with open(os.path.join(path, 'parameters'+str(i)+'.pkl'), 'wb') as f:
                pickle.dump(self.parameters[i], f, pickle.HIGHEST_PROTOCOL)


    def load_model(self, path:str) -> None:
        """
        Loads model from given path.
        """
        self.parameters = []
        parameter_paths = list(pathlib.Path(path).glob('parameters*.pkl'))

        for p in parameter_paths:
            with open(p, 'rb') as f:
                self.parameters.append(pickle.load(f))

    def reset_model(self) -> None:
        """
        Resets the model to blank
        """
        self.model = []
        self.parameters = []

    def feature_importance(self, path: str, X_test: np.array, in_var_names:list, out_var_names:list) -> None:
        assert X_test.shape[-1] == len(in_var_names)

        explainer = shap.Explainer(self.predict, X_test[:100])
        shap_values = explainer(X_test[:100]).values.mean(axis=0)
        
        if path is not None:
            figures_path = os.path.join(path, 'figures')
            if not os.path.exists(figures_path):
                os.makedirs(figures_path)

        plt.figure(figsize=(8, 6)) 
        plt.barh(np.arange(len(shap_values)), np.mean(shap_values, axis=-1))
        plt.yticks(np.arange(len(shap_values)), in_var_names)
        plt.title('Mean Shapley Values')
        plt.xlabel('Shapley values')
        plt.grid(True, which='major', color='#666666', linestyle='-')
        if path is not None:
            plt.savefig(os.path.join(figures_path, 'feature_importance_mean'), bbox_inches='tight', dpi=400)
        plt.show()
        plt.close()
            
        for i, ov in enumerate(out_var_names):
            plt.figure(figsize=(8, 8)) 
            plt.barh(np.arange(len(shap_values[:, i])), shap_values[:, i])
            plt.yticks(np.arange(len(shap_values[:, i])), in_var_names)
            plt.title(ov)
            plt.xlabel('Shapley values')
            plt.grid(True, which='major', color='#666666', linestyle='-')
            if path is not None:
                plt.savefig(os.path.join(figures_path, 'feature_importance_'+ov), bbox_inches='tight', dpi=400)
            plt.show()
            plt.close()
    

    def evaluate(self, path:str, X_test:np.array, y_test:np.array, in_var_names:list=None, out_var_names:list=None) -> None:
        """
        Evaluates the performance of the model.
        Computes training and test error, stores results and model in path.
        Generates diagonal matching figures and stores them in path.
        Plots feature importances.
        """

        print('evaluating on test set')
        predictions = self.predict(X_test)
        test_error  = np.mean((predictions - y_test)**2)

        print('appending to scoreboard')
        append_to_results(path, self.name, self.parameters, test_error)

        if in_var_names is None:
            in_var_names = IN_VAR_NAMES[1:]
        assert len(in_var_names) == X_test.shape[-1]

        if out_var_names is None:
            out_var_names = OUT_VAR_NAMES[1:]
        assert len(out_var_names) == y_test.shape[-1]

        figures_path = os.path.join(path, 'figures')
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        print('generating diagonal plots')
        for i, var in enumerate(out_var_names):
            accuracy_diagonal_plot(figures_path, X_test, y_test[:, i:i+1], predictions[:, i:i+1], var)

        mean_correction_factor_plot(y_test, predictions, figures_path, out_var_names)
        qq_plot(y_test, predictions, figures_path, out_var_names)
        ratio_plot(y_test, predictions, figures_path, out_var_names)

        print('gathering feature importances')
        self.feature_importance(path, X_test[:100], in_var_names, out_var_names)


