import os

import numpy as np
import tensorflow as tf

from models.MLModel import MLModel
from tensorflow.keras.layers import Input, Lambda
from utils import ReLoBRaLo, WarmUpLearningRateScheduler, det_loss, nonneg_loss


class NNModel(MLModel):
    """
    A parent class for representing a Neural Network Model
    """

    def __init__(self, name:str='NNModel') -> None:
        super().__init__(name)
        self.parameter_ranges = {
            'depth': (3, 10),
            'width': (32, 512),
            'activation': (0, 2.9),
            'learning_rate': (1e-5, 5e-3),
            'batch_size': (256, 512),
            'alpha': (-4, -1),
            'temperature': (-4, 1),
            'rho': (-5, -1)
        }

    
    def inner_train(self, learning_rate:float, batch_size:float, alpha:float, temperature:float, rho:float) -> float:
        weights = {'target_'+str(i): tf.keras.backend.variable(1.) for i in range(self.y_train.shape[1])}
        weights.update({'det': tf.keras.backend.variable(1.), 'nonneg': tf.keras.backend.variable(1.)})
        losses = {'target_'+str(i): 'mse' for i in range(self.y_train.shape[1])}
        losses.update({'det': det_loss, 'nonneg': nonneg_loss})
        def build_model(learning_rate:float) -> tf.keras.Model:
            x = Input((self.X_train.shape[1],))
            y = self.model[0](x)

            targets = [Lambda(lambda x: x, name='target_'+str(i))(y[:, i]) for i in range(self.y_train.shape[1])]
            det = Lambda(lambda x: x, name='det')(y[:, -9:])
            nonneg = Lambda(lambda x: tf.reduce_sum(x[0]*x[1]), name='nonneg')([x[:, -9:], y[:, -9:]])

            model = tf.keras.Model(x, targets+[det, nonneg])
            model.compile(
                loss=losses, 
                loss_weights=weights,
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
            return model

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=learning_rate*1e-3, verbose=1),
            #WarmUpLearningRateScheduler(3*self.X_train.shape[0]//batch_size, 1e-3, verbose=0),
            ReLoBRaLo(weights, alpha=1-(10**alpha), temperature=10**temperature, rho=1-(10**rho))
        ]

        model = build_model(learning_rate)

        model.fit(
            self.X_train, self.y_train, 
            batch_size=int(batch_size), 
            epochs=10000, 
            callbacks=callbacks,
            validation_data=(self.X_val, self.y_val),
            shuffle=True,
            verbose=True
        )

        tf.keras.backend.clear_session()
        return -np.mean((self.predict(self.X_val) - self.y_val)**2)


    def predict(self, X:np.array) -> np.array:
        return self.model[0].predict(X)


    def save_model(self, path:str) -> None:
        models_path = os.path.join(path, 'model')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        self.model[0].save(models_path)
        super().save_model(models_path)

    def load_model(self, path:str) -> None:
        model_path = os.path.join(path, 'model')
        self.model = [tf.keras.models.load_model(model_path)]
        super().load_model(model_path)
