# import numpy as np
# import tensorflow as tf
# from flwr_datasets import FederatedDataset
# import logging

# logging.basicConfig(level=logging.INFO)  # Configure logging
# logger = logging.getLogger(__name__)     # Create logger for the module

# class DataLoader:
#     """
#     Load federated dataset partition based on client ID and total number of clients
#     """
#     def __init__(self, client_id, total_clients):
#         self.client_id = client_id
#         self.total_clients = total_clients
#         self.data_sampling_percentage = None
#         self.x_train_sampled = None
#         self.y_train_sampled = None
#         self.load_and_partition_data()

#     def load_and_partition_data(self):
#         # fds = FederatedDataset(dataset="cifar10", partitioners={"train": self.total_clients})
#         fds = FederatedDataset(dataset="cifar10", partitioners={"train": 4})
#         partition = fds.load_partition(self.client_id-1, "train")
#         partition.set_format("numpy")
#         partition = partition.train_test_split(test_size=0.2)
#         # self.x_train, self.y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
#         # self.x_test, self.y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
#         # Original unfiltered training and test data
#         x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
#         self.x_test, self.y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
        
#         # Determine categories for each client
#         if self.client_id in [1, 2]:
#             categories = [0, 1]  # Categories for clients 1 and 2
#         elif self.client_id in [3, 4]:
#             categories = [2, 3, 4]  # Categories for clients 3 and 4
#         else:
#             logger.error("Invalid client ID, categories cannot be assigned.")
#             return

#         # Filter training data to only include specified categories
#         train_indices = np.isin(y_train, categories)
#         self.x_train, self.y_train = x_train[train_indices], y_train[train_indices]
#         # see the first 1000 rows of the training data
#         logger.info("Training data: %s", self.x_train[:500])

#     def load_train_data(self, data_sampling_percentage):
#         if data_sampling_percentage != self.data_sampling_percentage:
#             num_samples = int(data_sampling_percentage * len(self.x_train))
#             indices = np.random.choice(len(self.x_train), num_samples, replace=False)
#             self.x_train_sampled, self.y_train_sampled = self.x_train[indices], self.y_train[indices]
#             self.data_sampling_percentage = data_sampling_percentage
#         return (self.x_train_sampled, self.y_train_sampled)
    
#     def get_test_data(self):
#         return (self.x_test, self.y_test)

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
        self.x_test_sampled = None  # Add this line to store sampled test data
        self.y_test_sampled = None  # Add this line to store sampled test data
        self.load_and_partition_data()

    def load_and_partition_data(self):
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": 5})
        partition = fds.load_partition(self.client_id-1, "train")
        partition.set_format("numpy")
        partition = partition.train_test_split(test_size=0.2)
        x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
        self.x_test, self.y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
        
        if self.client_id in [1, 4]:
            categories = [0,1,2,3,4]  # Categories for clients 1 and 4
        elif self.client_id in [2, 3, 5]:
            categories = [0,1,2,3,4,5,6,7,8,9]  # Categories for clients 2 and 3
        else:
            logger.error("Invalid client ID, categories cannot be assigned.")
            return

        train_indices = np.isin(y_train, categories)
        self.x_train, self.y_train = x_train[train_indices], y_train[train_indices]

    def load_train_data(self, data_sampling_percentage):
        if data_sampling_percentage != self.data_sampling_percentage:
            num_samples = int(data_sampling_percentage * len(self.x_train))
            indices = np.random.choice(len(self.x_train), num_samples, replace=False)
            self.x_train_sampled, self.y_train_sampled = self.x_train[indices], self.y_train[indices]
            self.data_sampling_percentage = data_sampling_percentage
        return (self.x_train_sampled, self.y_train_sampled)
    
    def get_test_data(self):
        # Only sample if a new sampling percentage is provided and it's different from the current one
        if self.data_sampling_percentage != None:
            data_sampling_percentage = self.data_sampling_percentage
        else:
            data_sampling_percentage = 0.2
        
        data_sampling_percentage = max(data_sampling_percentage,0.2)
        
        num_samples = int(data_sampling_percentage * len(self.x_test))
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        self.x_test_sampled, self.y_test_sampled = self.x_test[indices], self.y_test[indices]
        return (self.x_test_sampled, self.y_test_sampled)
