import numpy as np
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf

class RandomizeCallback(Callback):

    def __init__(self, gates = [], switches = [], rate = 0.5):
        super(RandomizeCallback, self)
        self.gates = gates
        self.switches = switches
        self.rate = rate

    def random_switch(self):
        for gate in self.gates:
            if float(np.random.rand((1))) < self.rate:
                K.set_value(gate, 1)
            else:
                K.set_value(gate, 0)
        for switch in self.switches:
            if float(np.random.rand((1))) < self.rate:
                K.set_value(switch, 1)
            else:
                K.set_value(switch, 0)

    def on_train_begin(self, logs={}):
        for gate in self.gates:
                K.set_value(gate, 1)
        for switch in self.switches:
                K.set_value(switch, 1)

    def on_train_end(self, logs={}):
        for gate in self.gates:
                K.set_value(gate, 1)
        for switch in self.switches:
                K.set_value(switch, 1)

    def on_batch_end(self, batch, logs={}):
        self.random_switch()