import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as k
from keras.layers import LSTM,Dense,Flatten,Bidirectional
from keras.activations import softmax,relu,elu,sigmoid
from keras.optimizers import Adagrad
from keras.initializers import glorot_uniform   
from keras.regularizers import l2
from keras.constraints import min_max_norm
from keras.layers import Embedding,Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer
from keras.models import Sequential,Model
import MiniAttention

'''This shows an example to implement the MiniAttention Layer with Sequential and Model models in Keras.
    Compatible with Functional and Sequential Models of Keras.'''


def network(max_features):
    inp=Input(shape=(100,))
    z=Embedding(max_features,128)(inp)
    z=Bidirectional(LSTM(128))(z)
    z=MiniAttention.MiniAttentionBlock(keras.initializers.he_normal,None,None,None,None,None,None,None,None)(z)
    z=Dense(128,activation='relu')(z)
    z=Dense(1,activation='sigmoid')(z)
    model=Model(inputs=inp,outputs=z)
    model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='Adagrad')
    model.summary()
    return model

def network_sequential(max_features):
    model = Sequential()
    model.add(Embedding(max_features,128,input_shape=(100,)))
    model.add(MiniAttention.MiniAttentionBlock(None,None,None,None,None,None,None,None,None))
    model.add(LSTM(128))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='Adagrad')
    model.summary()
    return model

network(10000)
network_sequential(10000)