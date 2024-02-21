import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

class DataLoader:
    """
    Load federated dataset partition based on client ID and total number of clients
    """
    def __init__(self, client_id, total_clients):
        self.client_id = client_id
        self.total_clients = total_clients
        self.data_sampling_percentage = None
        self.x_train_sampled = None
        self.y_train_sampled = None
        self.load_and_partition_data()

    def load_and_partition_data(self):
        # fds = FederatedDataset(dataset="cifar10", partitioners={"train": self.total_clients})
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
        partition = fds.load_partition(self.client_id-1, "train")
        partition.set_format("numpy")
        partition = partition.train_test_split(test_size=0.2)
        self.x_train, self.y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
        self.x_test, self.y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    def load_train_data(self, data_sampling_percentage):
        if data_sampling_percentage != self.data_sampling_percentage:
            num_samples = int(data_sampling_percentage * len(self.x_train))
            indices = np.random.choice(len(self.x_train), num_samples, replace=False)
            self.x_train_sampled, self.y_train_sampled = self.x_train[indices], self.y_train[indices]
            self.data_sampling_percentage = data_sampling_percentage
        return (self.x_train_sampled, self.y_train_sampled)
    
    def get_test_data(self):
        return (self.x_test, self.y_test)