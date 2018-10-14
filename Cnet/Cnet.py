import math
import h5py

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, Reshape, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Concatenate, Add, Activation
from keras import backend as K
from keras_applications import inception_v3
from keras.engine.saving import load_attributes_from_hdf5_group, load_weights_from_hdf5_group_by_name

from .Select import Select
from .Bridge import Bridge


class Cnet(Model):
    def __repr__(self):
        print('Cnet')

    def __init__(self, inwidth=256, inheight=256, inchan=1, outwidth=256, outheight=256, classes=2, levels=2, depth=3,
                 base_filter=16, dropout=0.5, pretrained=False):
        inputs = Input(shape=(inwidth, inheight, inchan))
        scale_steps = inwidth // outwidth
        assert(inheight == scale_steps*outheight)

        self.levels = 2
        self.depth = 3
        self.base_filter = 32

        self.gates = []
        self.switches = []

        self.batchnorm = True
        self.dropout = dropout
        self.kernel_initializer = 'he_normal'
        self.activation = 'relu'
        self.pretrained = pretrained

        self.downblocks = []
        self.upblocks = []
        self.bridges = []
        self.b = self.block(self.base_filter//2, self.base_filter, inputs)
        for i in range(self.depth):
            input_filters = self.base_filter * 2**i
            output_filters = self.base_filter*2**(i+1)
            self.b = self.downblock(
                input_filters, output_filters, self.b)
            self.downblocks.append(self.b)

        self.bridge = self.downblock(
            self.base_filter*2**self.depth, self.base_filter*2**self.depth, self.b)

        m = self.base_filter * 2**(self.depth+1)
        for i in range(self.depth):
            input_filters = m // 2**i
            output_filters = m//2**(i+1)
            di = self.depth-i-1
            self.b = self.upblock(input_filters, output_filters, self.bridge)
            self.upblocks.append(self.b)
            gate = K.variable(1, dtype='uint8', name='gate')
            self.gates.append(gate)
            self.bridge = Bridge(gate)(
                [self.b, self.downblocks[di]])
            self.bridges.append(self.bridge)

        self.scale_block = self.upblock(
            self.base_filter*2, self.base_filter, self.bridge)

        for i in range(int(math.sqrt(scale_steps))):
            self.scale_block = self.downblock(
                self.base_filter//2**i, self.base_filter//2**(i+1), self.scale_block)

        self.out = Conv2D(classes, (1, 1), activation='sigmoid')(
            self.scale_block)

        super(Cnet, self).__init__(inputs=inputs, output=self.out)

        if self.pretrained == True:
            try:
                self.weights_path = keras.utils.get_file(
                    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                    inception_v3.WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
                self.load_pretrained_weights(self.weights_path)
            except Exception as e:
                print(e)

    def load_pretrained_weights(self, weights_path):
        f = h5py.File(self.weights_path, 'r')
        load_weights_from_hdf5_group_by_name(
            f, self.layers, skip_mismatch=True)

    def block(self, input_filters, output_filters, input_layer, down=True):
        layer = input_layer
        for i in range(self.levels) if down else reversed(range(self.levels)):
            layer = Conv2D(input_filters * 2**i, (3, 3), activation=self.activation, padding='same',
                           kernel_initializer=self.kernel_initializer)(layer)
        drop1 = Dropout(self.dropout)(layer)
        layer = Conv2D(output_filters, (3, 3), activation=self.activation, padding='same',
                       kernel_initializer=self.kernel_initializer)(drop1)
        if self.batchnorm is True:
            return BatchNormalization()(drop1)
        else:
            return drop1

    def downblock(self, input_filters, output_filters, input_layer):
        b = self.resnet(input_filters, input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2))(b)
        out = Conv2D(output_filters, (3, 3),
                     padding='same', activation=self.activation)(pool1)
        return out

    def upblock(self, input_filters, output_filters, input_layer):
        b = self.block(input_filters, 2*output_filters,
                       input_layer, down=False)
        upsample = UpSampling2D(size=(2, 2))(b)
        out = Conv2D(output_filters, (3, 3),
                     padding='same', activation=self.activation)(upsample)
        return out

    def resnet(self, filters, input_layer):
        x = input_layer
        for i in range(self.levels):
            y = Conv2D(filters, (1, 1), strides=(1, 1),
                       padding='same')(x)
            y = BatchNormalization()(y)
            y = Activation(self.activation)(y)
            switch = K.variable(1, dtype='uint8', name='switch')
            self.switches.append(switch)
            x = Select(switch)([y, x])
        x = Add()([input_layer, x])
        x = Activation(self.activation)(x)
        return x
