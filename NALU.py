import numpy as np
import keras.backend as K
from keras.layers import *
from keras.initializers import *
from keras.models import *

# NAC_NALU
class NALU(Layer):
    def __init__(self, units, MW_initializer='glorot_uniform',
                 G_initializer='glorot_uniform', mode="NALU",
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.MW_initializer = initializers.get(MW_initializer)
        self.G_initializer = initializers.get(G_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='M_hat')
        if self.mode == "NALU":
            self.G = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.G_initializer,
                                     name='G')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)
        if self.mode == "NAC":
            output = a
        elif self.mode == "NALU":
            m = K.exp(K.dot(K.log(K.abs(inputs) + 1e-7), W))
            g = K.sigmoid(K.dot(K.abs(inputs), self.G))
            output = g * a + (1 - g) * m
        else:
            raise ValueError("Valid modes: 'NAC', 'NALU'.")
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'mode' : self.mode,
            'MW_initializer': initializers.serialize(self.MW_initializer),
            'G_initializer':  initializers.serialize(self.G_initializer)
        }
        return dict(list(config.items()))
