from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, Reshape


class Cnet(Model):
    def __repr__(self):
        print('Cnet')

    def __init__(self, inputs=Input(shape=(256, 256, 1)), classes=2, dropout=0.5 ):

        self.dropout = dropout
        self.block1 = self.block(32, 128,inputs)
        self.block2 = self.block(128, 256,self.block1)
        self.block3 = self.block(256, 64,self.block2)
        self.out = Conv2D(1, (1, 1), activation='sigmoid')(self.block2)
        super(Cnet, self).__init__(inputs=inputs, output=self.out)

    def block(self, input_filters, output_filters, input_layer):
        conv1 = Conv2D(input_filters, (3,3), activation='relu', padding='same',
                       kernel_initializer='he_normal')(input_layer)
        conv2 = Conv2D(input_filters*2, (3,3), activation='sigmoid', padding='same')(conv1)
        conv3 = Conv2D(input_filters*4, (3,3), activation='sigmoid', padding='same')(conv2)
        out = Conv2D(output_filters, (1, 1), activation='sigmoid')(conv3)
        return out