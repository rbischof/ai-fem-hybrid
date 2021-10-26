import tensorflow as tf

from models.NNModel import NNModel
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout

class FCNNModel(NNModel):
    def __init__(self, name:str='FCNNModel') -> None:
        super().__init__(name)

    def inner_train(self, depth:float, width:float, activation:float, learning_rate:float, batch_size:float, alpha:float, temperature:float, rho:float) -> float:
        def build_model(depth:int, width:int, activation:str) -> tf.keras.Model:
            x = Input((self.X_train.shape[1],))
            f = Dense(width, activation=activation, name='f0')(x)

            for i in range(max(0, depth-2)):
                f = Dense(width, activation=activation, name='f'+str(i+1))(f)
                if i % 2 == 0:
                    f = BatchNormalization()(f)
                    f = Dropout(.05)(f)

            f = Dense(width, activation=activation, name='f')(f)
            y = Dense(self.y_train.shape[1])(f)

            model = tf.keras.Model(x, y)
            model.summary()
            return model

        self.model = [build_model(int(depth), int(width), ['relu', 'tanh', 'sigmoid'][min(int(activation), 2)])]
        
        return super().inner_train(learning_rate, batch_size, alpha, temperature, rho)