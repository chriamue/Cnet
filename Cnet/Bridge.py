import numpy as np
from keras import backend as K
from keras.layers import Concatenate, Lambda
from keras.engine.topology import Layer
from keras import layers


class Bridge(Layer):
    """
    concats both layers if gate is open,
    replicates first layer if gate is closed
    source: https://github.com/dblN/stochastic_depth_keras/blob/master/train.py
    """

    def __init__(self, gate, axis=-1, **kwargs):
        super(Bridge, self).__init__(**kwargs)
        self.axis = axis
        self.gate = gate

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        self.k = self.add_weight(
            name='zeros',
            shape=(),
            initializer='zeros',
            dtype='float32',
            trainable=False,
        )
        super(Bridge, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def call(self, inputs, training=None):
        assert isinstance(inputs, list)
        assert(len(inputs) == 2)
        input_a, input_b = inputs
        return K.switch(self.gate, Concatenate()([input_a, input_b]), Concatenate()([input_a, input_a]))
