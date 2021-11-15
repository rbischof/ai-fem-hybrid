import tensorflow as tf

from models.NNModel import NNModel
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout

class ResNetModel(NNModel):
    def __init__(self, name:str='ResNetModel') -> None:
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

    def inner_train(self, depth:float, width:float, activation:float, learning_rate:float, batch_size:float, alpha:float, temperature:float, rho:float) -> float:
        def build_model(depth:int, width:int, activation:str) -> tf.keras.Model:
            x = Input((self.X_train.shape[1],))
            f2 = Dense(width, activation=activation, name='f2_0')(x)
            f1 = f2
            for i in range(depth):
                f2 = Dense(width, activation=activation, name='f2_'+str(i+1))(f2) + f1
                if i % 2 == 0:
                    f2 = BatchNormalization()(f2)
                    f2 = Dropout(.1)(f2)
                f1 = f2
            f2 = Dense(width, activation=activation, name='f2_out')(f2)
            y = Dense(self.y_train.shape[1])(f2)

            model = tf.keras.Model(x, y)
            model.summary()
            return model

        if len(self.model) == 0:
            self.model = [build_model(int(depth), int(width), ['relu', 'tanh', 'sigmoid'][min(int(activation), 2)])]
        
        return super().inner_train(learning_rate, batch_size, alpha, temperature, rho)