import os
import pickle
import numpy as np

from sklearn.svm import LinearSVR
from models.MLModel import MLModel

class SVMModel(MLModel):
    def __init__(self, name:str='SVMModel') -> None:
        super().__init__(name)
        self.parameter_ranges = {
            'C': (1e-2, 5),
            'tol': (1e-5, 1e-3),
            'loss': (0, 2)
        }

    def inner_train(self, C, tol, loss) -> float:
        self.model.append(
            LinearSVR(
                C=C, 
                tol=tol,
                loss=['epsilon_insensitive', 'squared_epsilon_insensitive'][int(loss)],
                verbose=False
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

