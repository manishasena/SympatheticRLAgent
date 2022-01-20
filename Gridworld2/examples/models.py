import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from keras.layers import Input, Dense,Flatten, Conv2D, Concatenate, BatchNormalization, MaxPool2D, Activation
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.activations import softmax, sigmoid
#from tensorflow.python.keras.layers.core import Activation

import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
#from keras import backend as K
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)


class DQN:

    def __init__(self, width, height, num_actions):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        x = self.inputs
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door, self.R_x, self.R_y, self.H_x, self.H_y])
        x = Dense(32, activation="relu")(x)
        x = Dense(16, activation="relu")(x)
        self.outputs = Dense(self.num_actions)(x)
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net')

        #Compile
        optimiser = Adam(lr = 0.0001)

        self.model.compile(optimizer=optimiser,loss='mse',metrics=['accuracy'])
        print(self.model.summary())

class DQN_Human:

    def __init__(self, width, height, num_actions):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        x = self.inputs
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door, self.R_x, self.R_y, self.H_x, self.H_y])
        x = Dense(32, activation="relu")(x)
        x = Dense(16, activation="relu")(x)

        ### before soft max
        self.bsm = Dense(self.num_actions)(x)
        self.model_bsm = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.bsm, name='q_net_bsm')
        self.outputs = Activation(softmax)(self.bsm)
        ###
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net_human')

        #Compile
        optimiser = Adam(lr = 0.0001)

        self.model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['accuracy'])
        print(self.model.summary())

class nextStateModel:

    def __init__(self, width, height):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.out_shape = width*height + 5

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        self.action = Input(shape=(1,))

        x = self.inputs
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door, self.R_x, self.R_y, self.H_x, self.H_y])
        x = Concatenate(axis = 1)([x, self.action])
        x = Dense(32, activation="relu")(x)
        x = Dense(16, activation="relu")(x)

        ### before soft max
        x = Dense(self.out_shape)(x)
        self.outputs = Activation(sigmoid)(x)
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y, self.action], self.outputs, name='nextmodel')

        #Compile
        optimiser = Adam(lr = 0.0001)

        self.model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['accuracy'])
        print(self.model.summary())
