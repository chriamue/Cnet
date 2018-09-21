import numpy as np
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf

class SummaryCallback(Callback):

    def __init__(self, trainer, validation=False):
        super(SummaryCallback, self)
        self.trainer = trainer
        self.summarysteps = trainer.config['summarysteps']
        self.validation = validation
        self.image = tf.Variable(0., validate_shape=False)
        self.mask = tf.Variable(0., validate_shape=False)
        self.predicted = tf.Variable(0., validate_shape=False)
        model = self.trainer.model.model
        self.fetches = [tf.assign(self.image, model.inputs[0], validate_shape=False),
           tf.assign(self.mask, model.targets[0], validate_shape=False),
           tf.assign(self.predicted, model.outputs[0], validate_shape=False)]
        model._function_kwargs = {'fetches': self.fetches}

    def on_train_begin(self, logs={}):
        self.losses = []
        model = self.trainer.model.model
        self.fetches = [tf.assign(self.image, model.inputs[0], validate_shape=False),
           tf.assign(self.mask, model.targets[0], validate_shape=False),
           tf.assign(self.predicted, model.outputs[0], validate_shape=False)]
        model._function_kwargs = {'fetches': self.fetches}

    def on_train_end(self, logs={}):
        model = self.trainer.model.model
        model._function_kwargs = {'fetches': []}

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.losses.append(loss)
        if self.validation is False:
            self.trainer.global_step += 1
            if batch % self.summarysteps == 0:
                if self.trainer.summarywriter:
                    self.trainer.summarywriter.add_scalar(
                        self.trainer.name+'loss', loss, global_step=self.trainer.global_step)
                    image = K.eval(self.image)
                    if not type(image) is np.float32:
                        image = image[0]
                        image = np.rollaxis(image, axis=2, start=0)
                        mask = K.eval(self.mask)[0]
                        mask = np.rollaxis(mask, axis=2, start=0)
                        predicted = K.eval(self.predicted)[0]
                        predicted = np.rollaxis(predicted, axis=2, start=0)
                        self.trainer.summarywriter.add_image(
                                self.trainer.name+'image',image/255.0, global_step=self.trainer.global_step)
                        self.trainer.summarywriter.add_image(
                                self.trainer.name+'mask', mask.astype(np.float32), global_step=self.trainer.global_step)
                        self.trainer.summarywriter.add_image(
                                self.trainer.name+'predicted', predicted, global_step=self.trainer.global_step)
        else:
            if self.trainer.summarywriter:
                self.trainer.summarywriter.add_scalar(
                    self.trainer.name+'val_loss', loss, global_step=self.trainer.global_step)
                image = K.eval(self.image)
                if not type(image) is np.float32:
                    image = image[0]
                    image = np.rollaxis(image, axis=2, start=0)
                    mask = K.eval(self.mask)[0]
                    mask = np.rollaxis(mask, axis=2, start=0)
                    predicted = K.eval(self.predicted)[0]
                    predicted = np.rollaxis(predicted, axis=2, start=0)
                    self.trainer.summarywriter.add_image(
                            self.trainer.name+'val_image',image/255.0, global_step=self.trainer.global_step)
                    self.trainer.summarywriter.add_image(
                            self.trainer.name+'val_mask', mask, global_step=self.trainer.global_step)
                    self.trainer.summarywriter.add_image(
                            self.trainer.name+'val_predicted', predicted, global_step=self.trainer.global_step)
