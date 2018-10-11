# source: https://github.com/keras-team/keras/issues/3611

import numpy as np
from keras import backend as K

def dice(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def dice_crossentropy(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) * dice_loss(y_true, y_pred) 