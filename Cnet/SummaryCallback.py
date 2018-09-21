import numpy as np
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
class SummaryCallback(Callback):

    def __init__(self, trainer):
        self.trainer = trainer
        self.summarysteps = trainer.config['summarysteps']
        self.image = tf.Variable(0., validate_shape=False)
        self.mask = tf.Variable(0., validate_shape=False)

        model = self.trainer.model.model
        fetches = [tf.assign(self.image, model.inputs[0], validate_shape=False),
           tf.assign(self.mask, model.targets[0], validate_shape=False)]
        model._function_kwargs = {'fetches': fetches}
        super(SummaryCallback, self)

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        if batch > 35:
            self.ended = True
        loss = logs.get('loss')
        self.losses.append(loss)
        self.trainer.global_step += 1
        if batch % self.summarysteps == 0:
            print(self.trainer.global_step, batch, 'loss:', loss)
            if self.trainer.summarywriter:
                self.trainer.summarywriter.add_scalar(
                    self.trainer.name+'loss', loss, global_step=self.trainer.global_step)
                image = K.eval(self.image)
                if not type(image) is np.float32:
                    image = image[0]
                    image = np.rollaxis(image, axis=2, start=0)
                    mask = K.eval(self.mask)[0]
                    mask = np.rollaxis(mask, axis=2, start=0)
                    self.trainer.summarywriter.add_image(
                            self.trainer.name+'image',image/255.0, global_step=self.trainer.global_step)
                    self.trainer.summarywriter.add_image(
                            self.trainer.name+'mask', mask, global_step=self.trainer.global_step)
