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



class Model():
    def __init__(self, client_id, dpsgd: bool = False, learning_rate=0.2, l2_norm_clip=1.5, noise_multiplier=1.5, num_microbatches=1, delta=1e-5):
        self.learning_rate = learning_rate
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.delta = delta
        self.client_id = client_id
        self.dpsgd = False
        self.eval_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = tf.keras.applications.MobileNetV2((32, 32, 3), alpha=0.1, classes=10, weights=None)
        self.init_optimizer()
        self.detailed_data = pd.DataFrame(columns=['Epoch', 'Epsilon'])

    def init_optimizer(self):
        if self.dpsgd:
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


    def set_gradient_clipping(self, optimizer, clipvalue):
        if hasattr(optimizer, 'clipvalue'):
            optimizer.clipvalue = clipvalue
    
    def set_learning_rate(self, optimizer, learning_rate):
        tf.keras.backend.set_value(optimizer.learning_rate, learning_rate)


    def freeze_layers(self, freeze_percentage):
        if freeze_percentage == 0:
            for layer in self.model.layers:
                layer.trainable = True
        else:
            total_layers = len(self.model.layers)
            layers_to_freeze = int(total_layers * (freeze_percentage / 100))
            for layer in self.model.layers[:layers_to_freeze]:
                layer.trainable = False


    # create a function that returns the percentage of freezed layers in the model
    def get_freezed_layers_percentage(self):
        freezed_layers = 0
        for layer in self.model.layers:
            if not layer.trainable:
                freezed_layers += 1
        return freezed_layers / len(self.model.layers) * 100


    def apply_model_adjustments(self, config):
        """
        Applies the adjustments to the model based on the configuration.

        :param config: Configuration for the round.
        """
        self.set_learning_rate(self.optimizer, config["learning_rate"])
        self.set_gradient_clipping(self.optimizer, config.get("gradient_clipping_value"))
        self.freeze_layers(config.get("freeze_layers_percentage"))



    def fit(self, x_train, y_train, epochs, batch_size, config):
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
    
        # Prepare the training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

        row_index = 0

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
                        if not self.dpsgd:
                            loss = tf.reduce_mean(input_tensor=loss)
                        return loss
                
                    if self.dpsgd:
                        grads_and_vars = self.optimizer.compute_gradients(
                            loss_fn, var_list, gradient_tape=gradient_tape)
                    else:
                        grads_and_vars = self.optimizer.compute_gradients(loss_fn, var_list)

                self.optimizer.apply_gradients(grads_and_vars)

            # Compute the privacy budget expended per iteration
            if self.dpsgd:
                epsilon = self.compute_epsilon_per_epoch(epoch + 1, batch_size, len(x_train))
                self.detailed_data.loc[row_index] = [epoch+1, epsilon]
                row_index += 1

        
        # Save the data to a CSV file
        save_data_to_csv(f"mlflow/epsilon_data_client{self.client_id}.csv", self.detailed_data)   

        # Tests to check if the values were set correctly
        # assert self.optimizer.learning_rate == config["learning_rate"], "Learning rate was not set correctly"
        # if config.get("gradient_clipping_value"):
        #     assert self.optimizer.clipvalue == config["gradient_clipping_value"], "Gradient clipping value was not set correctly"
        # if config.get("freeze_layers_percentage"):
        #     assert self.get_freezed_layers_percentage() >= config["freeze_layers_percentage"] - 1 and self.get_freezed_layers_percentage() <= config["freeze_layers_percentage"] + 1, "Freeze layers percentage was not set correctly"
        return self.model
    

    