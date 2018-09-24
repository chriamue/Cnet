import numpy as np
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf

class RandomizeCallback(Callback):

    def __init__(self, gates = [], rate = 0.5):
        super(RandomizeCallback, self)
        self.gates = gates
        self.rate = rate

    def switch_gate(self):
        if float(np.random.rand((1))) < self.rate:
            for gate in self.gates:
                K.set_value(gate, 1)
        else:
            for gate in self.gates:
                K.set_value(gate, 0)

    def on_train_begin(self, logs={}):
        for gate in self.gates:
                K.set_value(gate, 1)

    def on_batch_end(self, batch, logs={}):
        self.switch_gate()