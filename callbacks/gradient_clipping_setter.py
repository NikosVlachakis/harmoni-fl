import tensorflow as tf

class GradientClippingCallback(tf.keras.callbacks.Callback):
    def __init__(self, clipvalue=1.0):
        self.clipvalue = clipvalue

    def on_train_batch_begin(self, batch, logs=None):
        if hasattr(self.model.optimizer, 'clipvalue'):
            self.model.optimizer.clipvalue = self.clipvalue
