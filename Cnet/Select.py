import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import layers


class Select(Layer):
    """
        Ignores second layer on given rate during training.
    # references
    - [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    """

    def __init__(self, switch, **kwargs):
        super(Select, self).__init__(**kwargs)
        self.switch = switch

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        self.k = self.add_weight(
            name='zeros',
            shape=(),
            initializer='zeros',
            dtype='float32',
            trainable=False,
        )
        super(Select, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a

    def call(self, inputs, training=None):
        assert isinstance(inputs, list)
        assert(len(inputs) == 2)
        input_a, input_b = inputs
        return K.switch(self.switch, input_a, input_b)
