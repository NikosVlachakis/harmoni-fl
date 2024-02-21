import os
import numpy as np
import pandas as pd
import tensorflow as tf
import dp_accounting
import logging
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
from utils.simple_utils import save_data_to_csv

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module


class FreezeLayersCallback(tf.keras.callbacks.Callback):
    def __init__(self, freeze_percentage):
        super().__init__()
        self.freeze_percentage = freeze_percentage

    def on_train_begin(self, logs=None):
        total_layers = len(self.model.layers)
        layers_to_freeze = int(total_layers * (self.freeze_percentage / 100.0))

        # Freeze the specified percentage of layers
        for layer in self.model.layers[:layers_to_freeze]:
            layer.trainable = False
        # Ensure the rest are trainable
        for layer in self.model.layers[layers_to_freeze:]:
            layer.trainable = True

        logger.info(f"FreezeLayersCallback: {layers_to_freeze} layers frozen.")


class LearningRateAdjustmentCallback(tf.keras.callbacks.Callback):
    def __init__(self, new_learning_rate):
        super().__init__()
        self.new_learning_rate = new_learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        # Access the model and its optimizer to adjust the learning rate
        if hasattr(self.model.optimizer, 'learning_rate'):
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.new_learning_rate)
            if epoch == 0:
                logger.info(f"Epoch {epoch+1}: Learning rate adjusted to {self.new_learning_rate}.")



class Model():
    def __init__(self, client_id, dp_opt: int = 0, learning_rate=0.2, l2_norm_clip=1.5, noise_multiplier=1.5, num_microbatches=1, delta=1e-5):
        self.learning_rate = learning_rate
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.delta = delta
        self.client_id = client_id
        self.dp_opt = dp_opt == 1
        self.eval_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = tf.keras.applications.MobileNetV2((32, 32, 3), alpha=0.1, classes=10, weights=None)
        # self.model = tf.keras.applications.EfficientNetB2(include_top=True, weights=None)

        self.init_optimizer()

    def init_optimizer(self):
        if self.dp_opt:
            logger.info("Using DPAdamGaussianOptimizer")
            # Differentially private optimizer
            self.optimizer = DPAdamGaussianOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=self.num_microbatches,
                learning_rate=self.learning_rate
            )
        else:
            logger.info("Using Adam Optimizer")
            # Optimizer for non-private training
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.eval_loss, metrics=["accuracy"])
        return self.model
    
    def get_model(self):
        return self.model
    
    def compute_epsilon_per_epoch(self, epoch, batch_size, number_of_examples):
        """Computes epsilon value at the end of each epoch."""
        if self.noise_multiplier == 0.0:
            return float('inf')
        # parameter that allows for a multi-faceted evaluation of privacy loss, ranging from average to worst-case scenarios
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        
        # keeps track of the overall privacy loss over multiple operations or steps in a data processing pipeline
        accountant = dp_accounting.rdp.RdpAccountant(orders)
        
        sampling_probability = batch_size / number_of_examples
        steps = epoch * (number_of_examples // batch_size)
        
        event = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(
                sampling_probability,
                dp_accounting.GaussianDpEvent(self.noise_multiplier)), steps)

        accountant.compose(event)
        return accountant.get_epsilon(self.delta)


    def set_learning_rate(self,model, learning_rate_value):
        """
        This function sets the learning rate of the optimizer.

        :param optimizer: The optimizer whose learning rate is to be set.
        :param learning_rate: The new learning rate.

        :return: None. The function modifies the optimizer in-place.
        """

        # Check if the optimizer has a 'learning_rate' attribute
        if hasattr(model.optimizer, 'learning_rate'):
            # If it does, set its learning rate to the new value
            tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate_value)


    def freeze_layers(self,model, freeze_percentage):
        """
        This function freezes a percentage of layers in the model.

        :param freeze_percentage: The percentage of layers to freeze. If it's 0, all layers are set to trainable.

        :return: None. The function modifies the model in-place.
        """

        # If freeze_percentage is 0, set all layers to trainable
        if freeze_percentage == 0:
            for layer in model.layers:
                layer.trainable = True
        else:
            # Calculate the total number of layers in the model
            total_layers = len(model.layers)

            # Calculate the number of layers to freeze based on the freeze_percentage
            layers_to_freeze = int(total_layers * (freeze_percentage / 100))

            # Freeze the first 'layers_to_freeze' layers
            for layer in model.layers[:layers_to_freeze]:
                layer.trainable = False



    def apply_model_adjustments(self, config):
        """
        Applies the adjustments to the model based on the configuration.

        :param config: Configuration for the round.
        """
        self.set_learning_rate(self.optimizer, config["learning_rate"])
        self.freeze_layers(config.get("freeze_layers_percentage"))


    def clip_gradients(self, grads_and_vars, clip_value):
        """
        This function clips the gradients by a specified value.

        :param grads_and_vars: List of tuples, each containing a gradient and its corresponding variable.
        :param clip_value: The threshold value for clipping. If it's 0, the gradients are not clipped.

        :return: List of tuples, each containing a clipped gradient and its corresponding variable.
        If clip_value is 0, it returns the input list as is.
        """

        # If clip_value is 0, return the input list as is
        if clip_value == 0:
            return grads_and_vars

        # Initialize the list for storing clipped gradients and variables
        clipped_grads_and_vars = []

        # Iterate over the gradients and variables
        for grad, var in grads_and_vars:
            # If the gradient is not None, clip it
            if grad is not None:
                clipped_grad = tf.clip_by_norm(grad, clip_value)
                clipped_grads_and_vars.append((clipped_grad, var))
            else:
                # If the gradient is None, append it as is
                clipped_grads_and_vars.append((grad, var))

        # Return the list of clipped gradients and variables
        return clipped_grads_and_vars


    def custom_fit(self, x_train, y_train, epochs, batch_size, config):
        """
        Custom training loop for the model with differential privacy.

        :param x_train: Features of the training dataset.
        :param y_train: Labels of the training dataset.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        :param config: Configuration for the round.
        """
    
        # Apply the model adjustments
        self.apply_model_adjustments(config)
        
        # Dataframe to store the detailed data and the index of the first row in the dataframe
        detailed_data = pd.DataFrame(columns=["epoch", "epsilon"])
        row_index = 0

        # Prepare the training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

        
        # Training loop
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape(persistent=True) as gradient_tape:
                    
                    # Compute logits and loss
                    logits = self.model(images, training=True)
                    var_list = self.model.trainable_variables

                    # In Eager mode, the optimizer takes a function that returns the loss.
                    def loss_fn():
                        logits = self.model(images, training=True) 
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels, logits=logits)  
                        # If training without privacy, the loss is a scalar not a vector.
                        if not self.dp_opt:
                            loss = tf.reduce_mean(input_tensor=loss)
                        return loss
                
                    if self.dp_opt:
                        grads_and_vars = self.optimizer.compute_gradients(
                            loss_fn, var_list, gradient_tape=gradient_tape)    

                    else:
                        grads_and_vars = self.optimizer.compute_gradients(loss_fn, var_list)
                    
                    # Clip gradients
                    grads_and_vars = self.clip_gradients(grads_and_vars, config.get("gradient_clipping_value"))
                    
                self.optimizer.apply_gradients(grads_and_vars)

            # Compute the privacy budget expended per iteration
            if self.dp_opt:
                epsilon = self.compute_epsilon_per_epoch(epoch + 1, batch_size, len(x_train))
                detailed_data.loc[row_index] = [epoch+1, epsilon]
                row_index += 1


        if self.dp_opt:
            # Save the data to a CSV file
            save_data_to_csv(f"mlflow/epsilon_data_client{self.client_id}.csv", detailed_data)   

        return self.model
    

   