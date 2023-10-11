import numpy as np
import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

def generate_data():
    if os.path.exists("./dataset/cifar10_data.npz"):
        data = np.load("./dataset/cifar10_data.npz")
        return {
            "x_train": data["x_train"],
            "y_train": data["y_train"],
            "x_test": data["x_test"],
            "y_test": data["y_test"],
        }
    else:
        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
        np.savez_compressed("./dataset/cifar10_data", 
                            x_train=x_train_full, y_train=y_train_full, 
                            x_test=x_test_full, y_test=y_test_full)
        return {
            "x_train": x_train_full,
            "y_train": y_train_full,
            "x_test": x_test_full,
            "y_test": y_test_full,
        }

import numpy as np

def load_data_helper(percentage=0.005):
    """simulate a partition."""
    
    data = generate_data()

    # Shuffle the data using a random seed
    indices = np.arange(len(data["x_train"]))
    np.random.shuffle(indices)

    # Determine the size of the subset based on the percentage
    subset_size = int(len(data["x_train"]) * percentage * np.random.randint(1, 5))

    x_train_subset = data["x_train"][indices[:subset_size]]
    y_train_subset = data["y_train"][indices[:subset_size]]

    
     # Shuffle the data using a random seed for x_test and y_test
    indices_test = np.arange(len(data["x_test"]))
    np.random.shuffle(indices_test)

    # Determine the size of the subset based on the percentage
    subset_size_test = int(len(data["x_test"]) * percentage  * np.random.randint(1, 5))

    x_test_subset = data["x_test"][indices_test[:subset_size_test]]
    y_test_subset = data["y_test"][indices_test[:subset_size_test]]


    logger.info("Loaded data. x_train shape: %s. y_train shape: %s. x_test shape: %s. y_test shape: %s", 
                x_train_subset.shape, y_train_subset.shape, x_test_subset.shape, y_test_subset.shape)
    
    return (x_train_subset, y_train_subset), (x_test_subset, y_test_subset)
