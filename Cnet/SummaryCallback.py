from keras.callbacks import Callback


class SummaryCallback(Callback):

    def __init__(self, trainer):
        self.trainer = trainer
        self.summarysteps = trainer.config['summarysteps']
        super(SummaryCallback, self)

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.losses.append(loss)
        self.trainer.global_step += 1
        if len(self.losses) % self.summarysteps == 0:
            print(self.trainer.global_step, batch, 'loss:', loss)
            if self.trainer.summarywriter:
                self.trainer.summarywriter.add_scalar(
                    self.trainer.name+'loss', loss, global_step=self.trainer.global_step)
