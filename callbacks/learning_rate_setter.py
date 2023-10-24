import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class LearningRateSetter(Callback):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.learning_rate = self.learning_rate
