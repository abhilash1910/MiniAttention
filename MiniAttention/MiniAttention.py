# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:45:35 2020

@author: Abhilash
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as k
from keras.layers import LSTM,Dense,Flatten,Bidirectional
from keras.activations import softmax,relu,elu,sigmoid
from tensorflow.keras.optimizers import Adagrad
from keras.initializers import glorot_uniform   
from keras.regularizers import l2
from keras.constraints import min_max_norm
from keras.layers import Embedding,Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer
from keras.models import Sequential,Model



def compute_dot(z,kernel):
    '''This is a simple dot product implementation with keras.backend'''
    return k.dot(z,kernel)


class MiniAttentionBlock(Layer):
    '''This is a Keras/Tensorflow implementation of Heirarchical Attention Networks for Document Classification (Yang etal,2015).
      Link:[https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf0]
    This is compatible with Keras and Tensorflow. 
    The input to this Layer should consist of 3 values-
     Input 3D Tensor - (samples,steps,features) 
    The output of this Layer consists of  2 values-
     Output 2D Tensor - (samples,features).
    This Layer can be used after the Keras.Embedding() Layer .
    This Layer can also be used on top of a LSTM/Bidirectional -LSTM/ GRU Layer ,return sequences should be 
    kept True.
    This Layer can also be used before the Dense Layer (after LSTM layers).
    It is recommended to use it either after the Embedding Layer or after the LSTM (recurrent) Layer and before the Dense Layer.
    '''
    def __init__(self,W_init,u_init,b_init,W_reg,u_reg,b_reg,W_const,u_const,b_const,bias=True):
        '''This initializes the weights and biases for the Attention Layer.
            The Weights have initializers,regularizers and constraints - denoted by W_<exp>
            where <exp> can be init,reg or const. These are consistent to be used with keras initializers,
            regularizers and constraints. The same is applied for bias and outputs (b and u).'''
        init_fn=keras.initializers.glorot_uniform
        self.W_init=W_init
        self.u_init=u_init
        self.b_init=b_init
        reg_fn= keras.regularizers.l2
        self.W_reg=W_reg
        self.u_reg=u_reg
        self.b_reg=b_reg
        const_fn=keras.constraints.min_max_norm
        self.W_const=W_const
        self.u_const=u_const
        self.b_const=b_const
        self.bias=bias
        super(MiniAttentionBlock,self).__init__()
    
    def attention_block(self,input_shape):
        '''This assigns the W,b and u with the values for Attention block.The Input of the Mini-Attention
            Block consists of 3D Tensor.'''
        assert(len(input_shape))==3
        self.W=self.add_weight((input_shape[-1],input_shape[-1],),initializer=self.W_init,regularizer=self.W_reg,constraint=self.W_const,name="Weight Layer")
        if self.bias==True:
            self.Bias= self.add_weight((input_shape[-1],),initializer=self.b_init,regularizer=self.b_reg,constraint=self.b_const,name="Bias Layer")
        self.u=self.add_weight((input_shape[-1],),initializer=self.u_init,regularizer=self.b_reg,constraint=self.b_const,name="Output Layer")
        super(MiniAttentionBlock,self).attention_block(input_shape)
        
    def build_nomask(self,inp):
        '''This implements the Un-masked Attention Layer.The weights are computed along with the biases (if any).
           Then the output is passed through a tanh activation.The weights are computed by dot product with the u layer.
           The corresponding outputs are passed through an exponential unit and then normalized. The final weights are 
           computed as a dot product of the input tensor and the weights.'''
        weights=compute_dot(inp,self.W)
        if self.bias==True:
            weights+=self.Bias
        #apply tanh
        weights=k.tanh(weights)
        f_weights=compute_dot(weights,self.u)
        f_weights=k.exp(f_weights)
        #normalize
        f_weights/=(k.sum(f_weights) + k.epsilon())
        #output
        f_weights=k.expand_dims(f_weights)
        output_weights=compute_dot(inp,f_weights)
        return k.sum(output_weights,axis=1)

