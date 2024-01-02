import numpy as np
import tensorflow as tf
import tensorflow_privacy
# from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer,DPKerasAdamOptimizer
import dp_accounting
import logging
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamOptimizer,DPAdamGaussianOptimizer

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module



class Model():
    def __init__(self, learning_rate=0.05, l2_norm_clip=1.5, noise_multiplier=1.5, num_microbatches=1, delta=1e-5):
        self.learning_rate = learning_rate
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.delta = delta
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = tf.keras.applications.MobileNetV2((32, 32, 3), alpha=0.1, classes=10, weights=None)

      
        # Differentially private optimizer
        self.dp_optimizer = DPAdamGaussianOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=self.num_microbatches,
            learning_rate=self.learning_rate
        )

        # Optimizer for non-private training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])
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


    def compute_epsilon_per_iteration(self,iteration, batch_size, number_of_examples):
        """Computes epsilon value at the end of each iteration."""
        if self.noise_multiplier == 0.0:
            return float('inf')
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        accountant = dp_accounting.rdp.RdpAccountant(orders)
        sampling_probability = batch_size / number_of_examples
        
        event = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(
                sampling_probability,
                dp_accounting.GaussianDpEvent(self.noise_multiplier)), iteration)

        accountant.compose(event)
        return accountant.get_epsilon(self.delta)



    def fit_method(self, train_dataset, epochs, batch_size, number_of_examples):
        final_epoch_accuracy = 0.0  # Variable to store the final epoch accuracy

        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            total_accuracy = 0
            total_batches = 0

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss = self.loss_function(y_batch_train, logits)
                
                # Compute DP gradients
                grads_and_vars = self.dp_optimizer.compute_gradients(loss, self.model.trainable_weights, tape=tape)

                # Apply DP gradients
                self.dp_optimizer.apply_gradients(grads_and_vars)

                # Calculate batch accuracy
                correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_batch_train, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                total_accuracy += accuracy.numpy()
                total_batches += 1

                # Log training process
                if step % 100 == 0:
                    loss_value = np.mean(loss.numpy())
                    logger.info(f"Epoch {epoch+1}, Step {step}, Loss: {loss_value}")

            # Compute and log epoch accuracy
            epoch_accuracy = total_accuracy / total_batches
            logger.info(f"End of epoch {epoch+1}, Accuracy: {epoch_accuracy}")

            # Compute epsilon at the end of the epoch
            epsilon = self.compute_epsilon_per_epoch(epoch+1, batch_size, number_of_examples)
            logger.info(f"End of epoch {epoch+1}, Epsilon: {epsilon}")

            if epoch == epochs - 1:  # If it's the last epoch, store the accuracy
                final_epoch_accuracy = epoch_accuracy

        return final_epoch_accuracy

