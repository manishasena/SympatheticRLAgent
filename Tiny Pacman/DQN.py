

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense,Flatten, Conv2D, Concatenate, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
import tensorflow as tf
#from keras import backend as K
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)


"""
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
"""

class DQN:
    def __init__(self,params):

        #self.input_shape = params['input_shape']
        self.HIDDEN_DIM = params['hidden_nodes']
        self.num_actions = params['number_actions']

        #self.inputs = Input(shape=(self.input_shape,))

        #input_length = params['window_size_H'] * params['window_size_W']

        self.inputs = Input(shape=(params['window_size_H'],params['window_size_W'],1,))
        self.power = Input(shape=(1,))
        self.x_self = Input(shape=(1,))
        self.y_self = Input(shape=(1,))
        self.x_other = Input(shape=(1,))
        self.y_other = Input(shape=(1,))

        x = Conv2D(16, 3, strides=(1, 1),padding = "valid", activation="relu")(self.inputs)
        x = Conv2D(32, 2, strides=(1, 1),padding = "valid", activation="relu")(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.power, self.x_self, self.y_self, self.x_other, self.y_other])
        x = Dense(128, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        self.outputs = Dense(self.num_actions)(x)
        self.model = Model([self.inputs, self.power, self.x_self, self.y_self, self.x_other, self.y_other], self.outputs, name='q_net')

        #Compile
        optimiser = Adam(lr = 0.0001)

        def my_loss_fn(y_true, y_pred):
            squared_difference = tf.reduce_sum(tf.square(y_true - y_pred),1, keepdims=True)
            return tf.reduce_mean(squared_difference)

        self.model.compile(optimizer=optimiser,loss='mse',metrics=['accuracy'])

        if params['load_file'] is not None:
            print('Loading checkpoint...')
            save_location = 'saves/' + params['load_file']
            self.model = load_model(save_location)
            #self.saver.restore(self.sess,self.params['load_file'])

        #Store loss over time
        self.training_loss = []

    def save_ckpt(self,filename):
        self.model.save(filename)


