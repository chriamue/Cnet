import math
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, Reshape, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Concatenate, Add, Activation


class Cnet(Model):
    def __repr__(self):
        print('Cnet')

    def __init__(self, inwidth=256, inheight=256, inchan=1, outwidth=256, outheight=256, classes=2, dropout=0.5):
        inputs = Input(shape=(inwidth, inheight, inchan))
        scale_steps = inwidth // outwidth
        assert(inheight == scale_steps*outheight)

        self.levels = 2
        self.depth = 5
        self.base_filter = 16

        self.batchnorm = True
        self.dropout = dropout
        self.kernel_initializer = 'he_normal'
        self.activation = 'relu'
        self.downblocks = []
        self.upblocks = []
        self.bridges = []
        self.b = self.block(self.base_filter//2, self.base_filter, inputs)
        for i in range(self.depth):
            self.b = self.downblock(self.base_filter * 2**i, self.base_filter*2**(i+1), self.b)
            self.downblocks.append(self.b)

        self.bridge = self.downblock(self.base_filter*2**self.depth, self.base_filter*2**self.depth, self.b)

        m = 16 * 2**self.depth
        for i in range(self.depth):
            self.b = self.upblock(m // 2**i, m//2**(i+1), self.bridge)
            self.upblocks.append(self.b)
            self.bridge = Concatenate()(
                [self.downblocks[self.depth-i-1], self.b])
            self.bridges.append(self.bridge)

        self.scale_block = self.upblock(self.base_filter*2, self.base_filter, self.bridge)

        for i in range(int(math.sqrt(scale_steps))):
            self.scale_block = self.downblock(
                self.base_filter//2**i, self.base_filter//2**(i+1), self.scale_block)

        self.out = Conv2D(1, (1, 1), activation='sigmoid')(self.scale_block)
        super(Cnet, self).__init__(inputs=inputs, output=self.out)

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
        for _ in range(self.levels):
            x = Conv2D(filters, (1, 1),strides=(1, 1),
                       padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            
        x = Add()([input_layer, x])
        x = Activation(self.activation)(x)
        return x
