import tensorflow as tf

from models.NNModel import NNModel
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate


class MaskNetModel(NNModel):
    def __init__(self, name:str='MaskNetModel') -> None:
        super().__init__(name)

    def inner_train(self, depth:float, width:float, activation:float, learning_rate:float, batch_size:float, alpha:float, temperature:float, rho:float) -> float:
        def model(depth:int, width:int, activation:str) -> tf.keras.Model:
            x = Input((self.X_train.shape[1],))
            f2 = Dense(width, activation=activation, name='f2_0')(x)
            f1 = f2
            for i in range(depth):
                if i % 2 == 0:
                    f20 = Dense(width, activation=activation, name='f20_'+str(i+1))(f2+f1)
                    f21 = Dense(self.X_train.shape[1], name='f21_'+str(i+1))(f2) * x
                    f22 = Dense(self.X_train.shape[1], name='f22_'+str(i+1))(f2) * (1 - x)
                    f2 = Concatenate()([f20, f21, f22])
                    f2 = BatchNormalization()(f2)
                    f2 = Dropout(.1)(f2)
                
                f2 = Dense(width, activation=activation, name='f2_'+str(i+1))(f2)
                f1 = f2

            f2 = Dense(width, activation=activation, name='f2_out')(f2)
            y = Dense(self.y_train.shape[1])(f2)

            model = tf.keras.Model(x, y)
            model.summary()
            return model

        self.model = [model(int(depth), int(width), ['relu', 'tanh', tf.nn.leaky_relu][min(int(activation), 2)])]
        
        return super().inner_train(learning_rate, batch_size, alpha, temperature, rho)