from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, Reshape, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization


class Cnet(Model):
    def __repr__(self):
        print('Cnet')

    def __init__(self, inputs=Input(shape=(256, 256, 1)), classes=2, dropout=0.5):

        self.dropout = dropout
        self.kernel_initializer = 'he_normal'
        self.activation = 'relu'
        self.block1 = self.block(4, 32, inputs)
        self.block2 = self.block(32, 128, self.block1)
        self.block3 = self.block(128, 32, self.block2)
        self.block4 = self.block(32, 4, self.block3)
        self.out = Conv2D(1, (1, 1), activation='sigmoid')(self.block4)
        super(Cnet, self).__init__(inputs=inputs, output=self.out)

    def block(self, input_filters, output_filters, input_layer):
        conv1 = Conv2D(input_filters, (3, 3), activation=self.activation, padding='same',
                       kernel_initializer=self.kernel_initializer)(input_layer)
        conv2 = Conv2D(input_filters*2, (3, 3), activation=self.activation,
                       padding='same', kernel_initializer=self.kernel_initializer)(conv1)
        conv3 = Conv2D(input_filters*4, (3, 3), activation=self.activation,
                       padding='same', kernel_initializer=self.kernel_initializer)(conv2)
        drop1 = Dropout(self.dropout)(conv3)
        bn = BatchNormalization()(drop1)
        return bn

    def downblock(self, input_filters, output_filters, input_layer):
        b = self.block(input_filters, 2*output_filters, input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2))(b)
        out = Conv2D(output_filters, (3, 3), activation=self.activation)(pool1)
        return out

    def upblock(self, input_filters, output_filters, input_layer):
        b = self.block(input_filters, 2*output_filters, input_layer)
        upsample = UpSampling2D(size=(2, 2))(b)
        out = Conv2D(output_filters, (3, 3), activation=self.activation)(upsample)
        return out
