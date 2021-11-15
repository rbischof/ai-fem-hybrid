import os
import pickle
import numpy as np

from models.MLModel import MLModel
from sklearn.neighbors import KNeighborsRegressor

class KNNModel(MLModel):
    def __init__(self, name:str='KNNModel') -> None:
        super().__init__(name)
        self.parameter_ranges = {
            'n_neighbors': (1, 10),
            'p': (1, 4)
        }


    def inner_train(self, n_neighbors, p) -> float:
        if len(self.model) == 0:
            self.model.append(
                KNeighborsRegressor(
                    n_neighbors=int(n_neighbors), 
                    p=int(p),
                    weights='distance',
                    n_jobs=-1
                )
            )

        self.model[0].fit(self.X_train, self.y_train)
        return -np.mean((self.predict(self.X_val) - self.y_val)**2)
        

    def save_model(self, path:str) -> None:
        models_path = os.path.join(path, 'model')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        pickle.dump(self.model[0], open(os.path.join(models_path, 'model.pkl'), 'wb'))
        super().save_model(models_path)

    def load_model(self, path:str) -> None:
        model_path = os.path.join(path, 'model', 'model.pkl')
        self.model = [pickle.load(open(model_path, 'rb'))]
        super().load_model(model_path)
