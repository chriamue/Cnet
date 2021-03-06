from __future__ import absolute_import
import os
import cv2
import numpy as np
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input
import tensorflow as tf

from protoseg.backends import AbstractBackend
from protoseg.predictor import Predictor
from tensorboardX import SummaryWriter

from .Cnet import Cnet
from .DiceLoss import *
from .RandomizeCallback import RandomizeCallback
from .SummaryCallback import SummaryCallback
K.set_image_data_format('channels_last')


class Cnet_backend(AbstractBackend):

    config = None

    def __init__(self):
        AbstractBackend.__init__(self)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        set_session(sess)

    def load_model(self, config, modelfile):
        self.config = config
        model = Cnet(config['width'], config['height'],
                     3 if config['color_img'] == True else 1, config['mask_width'], config['mask_height'],
                     classes=config['classes'], levels=config['cnet_levels'], depth=config['cnet_depth'],
                     base_filter=config['cnet_base_filter'], dropout=config['dropout'], pretrained=config['pretrained'])
        model.summary()
        if os.path.isfile(modelfile):
            model.load_weights(modelfile, by_name=True)
            print('loaded model from:', modelfile)
        return model

    def save_model(self, model):
        model.model.save_weights(model.modelfile)
        print('saved model to:', model.modelfile)

    def init_trainer(self, trainer):
        self.config = trainer.config
        optimizer = trainer.config['optimizer']
        loss = trainer.config['loss_function']
        if trainer.config['loss_function'] == 'default':
            loss = 'categorical_crossentropy'
        elif trainer.config['loss_function'] == 'dice_loss':
            loss = dice_loss
        elif trainer.config['loss_function'] == 'dice_crossentropy':
            loss = dice_crossentropy
        trainer.model.model.compile(optimizer=optimizer, loss=loss,
                                    metrics=['accuracy', dice])
        trainer.predictor = Predictor(
            model=trainer.model, config=trainer.config, backend=self)
        self.summary_callback = SummaryCallback(trainer)
        self.randomize_callback = RandomizeCallback(gates=trainer.model.model.gates, switches=trainer.model.model.switches,
                                                    rate=trainer.config['dropout'])

    def dataloader_format(self, img, mask=None):
        img = np.atleast_3d(img)
        if mask is None:
            return img

        if mask.ndim > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = np.atleast_3d(mask)
        mask[mask > 0] = 1  # binary mask
        return img, keras.utils.to_categorical(mask, num_classes=self.config.get('classes'))

    def datagenerator(self, generator):
        for (imgs, labels) in generator:
            yield np.array(imgs), np.array(labels)

    def train_epoch(self, trainer):
        print('train on Cnet backend')
        model = trainer.model.model
        batch_size = trainer.config['batch_size']
        summarysteps = trainer.config['summarysteps']

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=summarysteps,
                                                           batch_size=batch_size, write_graph=True, write_grads=False,
                                                           write_images=False)
        datagen = self.datagenerator(
            trainer.dataloader.batch_generator(batch_size, shuffle=True))
        val_x0, val_y0 = trainer.valdataloader[0]
        val_x1, val_y1 = trainer.valdataloader[1]
        model.fit_generator(datagen, steps_per_epoch=len(
            trainer.dataloader)//batch_size-1, epochs=1, validation_data=(np.array([val_x0, val_x1]), np.array([val_y0, val_y1])), validation_steps=2,
            callbacks=[self.summary_callback, self.randomize_callback])  # , tensorboard_callback])

    def validate_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        datagen = self.datagenerator(
            trainer.valdataloader.batch_generator(batch_size))
        global_step = int(
            trainer.epoch * len(trainer.valdataloader)/batch_size)
        for i, (X_batch, y_batch) in enumerate(datagen):
            X_batch = X_batch.astype(np.float32)
            prediction = trainer.predictor.predict(X_batch[0])
            y = np.argmax(y_batch[0], axis=-1).squeeze()
            trainer.metric(
                prediction, y, prefix=trainer.name)
            if trainer.summarywriter:
                image = (np.squeeze(X_batch[0])/255.0)
                mask = y
                predicted = prediction
                trainer.summarywriter.add_image(
                    trainer.name+"val_image", image, global_step=global_step+i)
                trainer.summarywriter.add_image(
                    trainer.name+"val_mask", mask, global_step=global_step+i)
                trainer.summarywriter.add_image(
                    trainer.name+"val_predicted", predicted, global_step=global_step+i)
                raw_prediction = self.predict(trainer.predictor, X_batch[0])
                raw_prediction = (np.sum(raw_prediction, axis=-1) - raw_prediction[:, :, 0])/(
                    raw_prediction.shape[-1]-1)  # mean of classes without background
                trainer.summarywriter.add_image(
                    trainer.name+"val_predicted_mean", raw_prediction, global_step=global_step+i)

    def get_summary_writer(self, logdir='results/'):
        self.logdir = logdir
        return SummaryWriter(log_dir=logdir)

    def predict(self, predictor, img):
        predict = self.batch_predict(predictor, np.array([img]))[0]
        return predict

    def batch_predict(self, predictor, img_batch):
        model = predictor.model.model
        predict = model.predict_on_batch(img_batch)
        return predict
