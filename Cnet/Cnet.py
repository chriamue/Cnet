from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D


class Cnet(Model):
    def __repr__(self):
        print('Cnet')

    def __init__(self, inputs=Input(shape=(256, 256, 1)), classes=2, dropout=0.5 ):

        self.dropout = dropout
        conv1 = Conv2D(64, (3,3), activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv2 = Conv2D(64, (3,3), activation='sigmoid', padding='same')(conv1)
        d1 = Dense( classes , activation='softmax', name='predictions')(conv2)
        output = Conv2D(1, (1, 1), activation='sigmoid') (d1)
        super(Cnet, self).__init__(inputs=inputs, output=output)
