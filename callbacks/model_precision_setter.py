import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class ModelPrecisionAdjustmentCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_dtype='float32'):
        if target_dtype not in ['float16', 'float32', 'float64']:
            raise ValueError(f"Invalid target data type: {target_dtype}. Valid options are 'float16', 'float32', 'float64'.")
        self.target_dtype = target_dtype
        self.original_model = None

    def on_train_begin(self, logs=None):
        # Check if the current model's data type already matches the target data type
        current_dtype = self.model.dtype
        if current_dtype == self.target_dtype:
            return
        
        self.original_model = self.model
        self.model = self.convert_model_dtype(self.model, self.target_dtype)

    def on_train_end(self, logs=None):
        if self.original_model is not None:
            self.model.set_weights(self.original_model.get_weights())
            logger.info("Restored original model precision")

    @staticmethod
    def convert_model_dtype(model, target_dtype):
        """Convert the data type of a model's weights."""
        # Clone the model to ensure we're not modifying the original model
        with tf.keras.utils.custom_object_scope({}):
            new_model = tf.keras.models.clone_model(model)
        
        # Build the model if not built
        if not new_model.built:
            new_model.build(input_shape=model.input_shape)
        
        # Convert the weights of each layer to the target dtype
        for layer in new_model.layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer.kernel_initializer, 'dtype'):
                layer.kernel_initializer.dtype = target_dtype
            if layer.weights:
                converted_weights = [w.numpy().astype(target_dtype) for w in layer.weights]
                layer.set_weights(converted_weights)
        
        return new_model
