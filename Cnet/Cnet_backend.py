from __future__ import absolute_import
import os
import cv2
import numpy as np
import keras
from keras.layers import Input
from protoseg.backends import AbstractBackend
from tensorboardX import SummaryWriter

from .Cnet import Cnet
from .SummaryCallback import SummaryCallback


class Cnet_backend(AbstractBackend):

    def __init__(self):
        AbstractBackend.__init__(self)

    def load_model(self, config, modelfile):
        inputs = Input(shape=(config['width'], config['height'],
                              3 if config['color_img'] == True else 1))
        model = Cnet(
            inputs=inputs, classes=config['classes'], dropout=config['dropout'])
        if os.path.isfile(modelfile):
            model.load_weights(modelfile, by_name=True)
            print('loaded model from:', modelfile)
        return model

    def save_model(self, model):
        model.model.save_weights(model.modelfile)
        print('saved model to:', model.modelfile)

    def init_trainer(self, trainer):
        optimizer = trainer.config['optimizer']
        loss = trainer.config['loss_function']
        if trainer.config['loss_function'] == 'default':
            loss = 'categorical_crossentropy'
        trainer.model.model.compile(optimizer=optimizer, loss=loss)

    def dataloader_format(self, img, mask=None):
        if img.ndim < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.atleast_3d(img)
        if mask is None:
            return img

        if mask.ndim > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = np.atleast_3d(mask)
        mask[mask > 0] = 1  # binary mask
        return img, mask

    def datagenerator(self, generator):
        for (imgs, labels) in generator:
            yield np.array(imgs), np.array(labels)

    def train_epoch(self, trainer):
        print('train on gluoncv backend')
        batch_size = trainer.config['batch_size']
        summarysteps = trainer.config['summarysteps']
        summary_callback = SummaryCallback(trainer)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=0, batch_size=summarysteps, write_graph=True, write_grads=False,
                                                           write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                                                           embeddings_data=None)
        datagen = self.datagenerator(
            trainer.dataloader.batch_generator(batch_size))
        trainer.model.model.fit_generator(datagen, steps_per_epoch=len(
            trainer.dataloader)//batch_size, epochs=1, callbacks=[summary_callback, tensorboard_callback])

    def validate_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        datagen = self.datagenerator(
            trainer.valdataloader.batch_generator(batch_size))
        hist = trainer.model.model.fit_generator(datagen, steps_per_epoch=len(
            trainer.valdataloader)//batch_size, epochs=1, verbose=1, callbacks=None,
            validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1,
            use_multiprocessing=False, shuffle=True, initial_epoch=0)
        print(hist)

    def get_summary_writer(self, logdir='results/'):
        self.logdir = logdir
        return SummaryWriter(log_dir=logdir)

    def predict(self, predictor, img):
        img_batch = np.array([img])
        return self.batch_predict(predictor, img_batch)

    def batch_predict(self, predictor, img_batch):
        model = predictor.model.model
        predict = model.predict_on_batch(img_batch)
        return predict
