import numpy as np
from keras import backend as K
from keras.layers import Concatenate
from keras.engine.topology import Layer
from keras import layers


class Bridge(Layer):
    """
    """

    def __init__(self, rate=0.5, axis=-1, **kwargs):
        super(Bridge, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.axis = axis

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
        return Concatenate()([input_a, input_b])
