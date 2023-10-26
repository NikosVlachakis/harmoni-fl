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


def load_data_helper(percentage=0.05, batch_size=32):
    data = generate_data()

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((data["x_train"], data["y_train"]))
    test_dataset = tf.data.Dataset.from_tensor_slices((data["x_test"], data["y_test"]))

    # Calculate subset size for train and test datasets
    train_subset_size = int(len(data["x_train"]) * percentage)
    test_subset_size = int(len(data["x_test"]) * percentage)

    # Shuffle and subset data
    train_dataset = train_dataset.shuffle(buffer_size=len(data["x_train"])).take(train_subset_size)
    test_dataset = test_dataset.shuffle(buffer_size=len(data["x_test"])).take(test_subset_size)

    # Batch data
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    logger.info("Created data generators with batch size and subset size: %s, %s", batch_size, percentage)
    
    return train_dataset, test_dataset, train_subset_size, test_subset_size
