import os
import csv
import numpy as np
import tensorflow as tf

from time import time, strftime, gmtime

IN_VAR_NAMES  = ['ID']+['Reinforcing Layer']*5+['Material Model Nr.']+['CMM Usermat Model']*6 + ['Reinforcement Area',
                'Reinforcement Diameter','Effective Reinforcement TCM','Yield Stress Reinforcement',
                'Ultimate Stress Reinforcement','Ultimate Strain Reinforcement','Reinforcement Angle',
                'Concrete Compression Strength','Ultimate Strain Concrete','Normal Strain X',
                'Normal Strain Y','Shear Strain XY']

OUT_VAR_NAMES = ['ID', 'Normal Stress in X-Direction $\sigma_x$', 'Normal Stress in Y-Direction $\sigma_y$',
                'Normal Stress in XY-Direction $\sigma_{xy}$', 'Stiffness Tensor Component $K_{11}$',
                'Stiffness Tensor Component $K_{12}$', 'Stiffness Tensor Component $K_{13}$',
                'Stiffness Tensor Component $K_{21}$', 'Stiffness Tensor Component $K_{22}$', 
                'Stiffness Tensor Component $K_{23}$', 'Stiffness Tensor Component $K_{31}$',
                'Stiffness Tensor Component $K_{32}$', 'Stiffness Tensor Component $K_{33}$']


def create_directory(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        ix = 1
        if path[-1] == '/':
            path = path[:-1]
        alternative_path = path + '_' + str(ix) + '/'
        while os.path.exists(alternative_path):
            ix += 1
            alternative_path = path + '_' + str(ix) + '/'
        path = alternative_path
        os.makedirs(path)
    return path


def append_to_results(path:str, name:str, parameters, val_error:float):
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    
    if tf.is_tensor(val_error):
        val_error = val_error.numpy()

    output = [strftime('%d.%m. %H:%M:%S', gmtime(time())), path, name, val_error, parameters]

    try:
        if not os.path.exists('experiments'):
            with open('experiments/results.csv', 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'path', 'model_name', 'val_error', 'parameters'])
        with open('experiments/results.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(output)
    except IOError:
        print('I/O error')


class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

class ReLoBRaLo(tf.keras.callbacks.Callback):
    def __init__(self, weighting:dict, alpha:float, temperature:float, rho:float):
        self.weighting = weighting
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.lambdas = [tf.constant(1.)]*len(weighting)
        self.losses = [tf.constant(1.)]*len(weighting)
        self.init_loss = [tf.constant(1.)]*len(weighting)
        self.batch_count = 0
    
    def on_train_batch_begin(self, batch:int, logs:dict={}):
        # set lambdas computed in last batch
        for v, l in zip(list(self.weighting.values()), self.lambdas):
            tf.keras.backend.set_value(v, l)

    def on_train_batch_end(self, batch:int, logs:dict={}):
        # reset all weights to 1 for validation
        for w in self.weighting.values():
            tf.keras.backend.set_value(w, 1.)

        # prepare lambdas for next batch
        # find losses in logs or raise error
        losses = []
        for k in self.weighting.keys():
            loss = logs.get(k+'_loss')
            if loss is None:
                pass
                # print(k, 'not in logs:', logs)
            else:
                losses.append(loss)

        # in first iteration, drop lambda_hat and use init lambdas, i.e. lambda = 1
        if self.batch_count == 0:
            alpha = 1.
            rho = 1.
        # in second iteration, drop init lambdas and use only lambda_hat
        elif self.batch_count == 1:
            alpha = 0.
            rho = 1.
        # in following iterations, default behaviour
        else:
            alpha = self.alpha
            rho = (np.random.uniform(size=1) < self.rho).astype(np.float32)[0]

        lambdas_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(self.losses[i]*self.temperature+1e-7) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
        init_lambdas_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(self.init_loss[i]*self.temperature+1e-7) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
        self.lambdas = [rho*alpha*self.lambdas[i] + (1-rho)*alpha*init_lambdas_hat[i] + (1-alpha)*lambdas_hat[i] for i in range(len(losses))]

        # in first iteration, store losses in init_loss
        if self.batch_count == 0:
            self.init_loss = losses
        self.losses = losses
        self.batch_count += 1

    def on_epoch_end(self, epoch:int, logs:dict={}):
        # add lambdas to log
        logs['lambdas'] = [l.numpy() for l in self.lambdas]
        print('lambdas', logs['lambdas'])


def det_loss(y, pred):
    return tf.reduce_mean((tf.linalg.det(tf.reshape(y[:, -9:], (-1, 3, 3))+1e-3) - tf.linalg.det(tf.reshape(pred[:, -9:], (-1, 3, 3))*1e-3))**2)

def nonneg_loss(y, pred):
    return tf.nn.relu(-pred)