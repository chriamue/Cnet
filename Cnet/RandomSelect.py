import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import layers


class RandomSelect(Layer):
    """
        Ignores second layer on given rate during training.

    # references
    - [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    """

    def __init__(self, rate=0.5, **kwargs):
        super(RandomSelect, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        self.k = self.add_weight(
            name='zeros',
            shape=(),
            initializer='zeros',
            dtype='float32',
            trainable=False,
        )
        super(RandomSelect, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a

    def call(self, inputs, training=None):
        assert isinstance(inputs, list)
        assert(len(inputs) == 2)
        input_a, input_b = inputs
        if 0. < self.rate < 1.:
            if float(np.random.rand(1)) > self.rate:
                return layers.Add()([
                    input_b,
                    K.tf.multiply(self.k, input_a),
                ])
        return layers.Add()([
            input_a,
            K.tf.multiply(self.k, input_b),
        ])
