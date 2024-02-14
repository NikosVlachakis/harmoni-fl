import time
import os
import argparse
import flwr as fl
from services.prometheus_queries import *
from services.prometheus_service import PrometheusService
import tensorflow as tf
import logging
from helpers.load_data import DataLoader
import os
import numpy as np
from helpers.mlflow import MlflowHelper
from model.model import Model
from model.sparsification import Sparsifier

from utils.simple_utils import calculate_weights_size


logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flower client')

parser.add_argument('--server_address', type=str, default="server:8080", help="Address of the server")
parser.add_argument('--client_id', type=int, default=1, help="Unique ID for the client")
parser.add_argument('--total_clients', type=int, default=2, help="Total number of clients")
parser.add_argument('--dp_opt', type=int, default=0, help="dp_opt or not (default: 0)")

args = parser.parse_args()

# Create an instance of the model
model_instance = Model(client_id=args.client_id, dp_opt=args.dp_opt)
class Client(fl.client.NumPyClient):
    def __init__(self):
        self.round_metrics = {}
        self.properties = {}
        self.prom_service = PrometheusService()
        self.mlflow_helper = MlflowHelper(client_id=args.client_id, dp_opt=args.dp_opt)
        self.container_name = os.getenv('container_name')
        self.properties.update({
            "container_name": self.container_name
        })
        self.args = args
        self.data_loader = DataLoader(client_id=self.args.client_id, total_clients=self.args.total_clients)       
     


    def get_parameters(self, config):
        return model_instance.get_model().get_weights()

    def fit(self, parameters, config):
        
        # Load the data 
        (x_train, y_train) = self.data_loader.load_train_data(data_sampling_percentage=config["data_sample_percentage"])
       
        start_time = time.time()  # Capture the start time
       
        # Set the weights of the model
        model_instance.get_model().set_weights(parameters)

        # Update the properties with the config as it contains the new configuration for the round
        self.properties.update(config)

        logger.info("container_name is: %s",  self.container_name)
        logger.info("config is: %s", config)
        

        # Train the model
        model_instance.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], config=config)      

        # Get the weights after training
        parameters_prime = model_instance.get_model().get_weights()

        end_time = time.time()  # Capture the end time
        duration = end_time - start_time  # Calculate duration

        # Store operational metrics for fit
        self.round_metrics.update({
            "fit_start_time": start_time,
            "fit_end_time": end_time,
            "fit_duration": duration
        })

        # Calculate evaluation metric
        results = {
            "start_time": start_time,
            "end_time": end_time,
            "container_name": self.container_name,
        }     
       
        # Check if sparsification is enabled
        # if config["sparsification_enabled"]:
                        
        #     # Create a sparsifier object 
        #     sparsifier = Sparsifier(method=config["sparsification_method"], percentile=config["sparsification_percentile"])
            
        #     serialized_sparse_weights, total_nnz = sparsifier.sparsify_and_serialize_weights(parameters_prime)

        #     # Get the size of the serialized sparse weights and the original weights
        #     serialized_sparse_weights_size = calculate_weights_size(serialized_sparse_weights)
        #     original_weights_size = calculate_weights_size(parameters_prime)    
            
        #     self.round_metrics.update({
        #         "total_nnz": total_nnz,
        #         "serialized_sparse_weights_size": serialized_sparse_weights_size,
        #         "original_weights_size": original_weights_size
        #     })

        #     # Return new weights, number of training examples, and results
        #     return serialized_sparse_weights, len(x_train), results

        # else:
        #     # Directly return the dense weights without sparsification
        #     return parameters_prime, len(x_train), results
        
        return parameters_prime, len(x_train), results
        

    def evaluate(self, parameters, config):
        
        (x_test, y_test) = self.data_loader.get_test_data()

        start_time = time.time()  # Capture the start time
        
        model_instance.get_model().set_weights(parameters)

        compiled_model = model_instance.compile()

        # Evaluate the model and get the loss and accuracy
        loss, accuracy = compiled_model.evaluate(x_test, y_test)

        end_time = time.time()  # Capture the end time
        duration = end_time - start_time  # Calculate duration

        # Store operational metrics for evaluation
        self.round_metrics.update({
            "eval_start_time": start_time,
            "eval_end_time": end_time,
            "eval_duration": duration,
            "test_set_size": len(x_test),
            "test_set_accuracy": accuracy
        })

        self.mlflow_helper.log_round_metrics_for_client(
            container_name=os.getenv('container_name'),
            server_round=config["server_round"],
            experiment_id=config["experiment_id"],
            operational_metrics={**self.round_metrics},
            properties=self.properties
        )

        return float(loss), len(x_test), {"accuracy": float(accuracy)}
    

    def get_properties(self, *args, **kwargs):
            return self.properties

    
def start_fl_client():
    try:
        fl.client.start_client(server_address=args.server_address, client=Client().to_client())
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    start_fl_client()
