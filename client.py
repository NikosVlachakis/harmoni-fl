import time
import os
import argparse
from flask import Flask
from flask_cors.extension import CORS
from models.callbacks.gradient_clipping_setter import GradientClippingCallback
from models.callbacks.model_precision_setter import ModelPrecisionAdjustmentCallback
from flask_restful import Resource, Api
import flwr as fl
from models.model_adjustments import ModelAdjuster
from services.prometheus_queries import *
from services.prometheus_service import PrometheusService
import tensorflow as tf
from threading import Thread
import logging
from helpers.load_data import DataLoader
import os
import numpy as np
from helpers.mlflow import MlflowHelper
from models.model import Model
from models.callbacks.learning_rate_setter import LearningRateSetter
from models.sparsification import Sparsifier

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

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument
model = Model()

# Compile the model
model.compile()

# Get the model
model = model.get_model()

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.fit_operational_metrics = {}
        self.eval_operational_metrics = {}
        self.properties = {}
        self.prom_service = PrometheusService()
        self.container_name = os.getenv('container_name')
        self.properties.update({
            "container_name": self.container_name
        })
        self.args = args
        self.data_loader = DataLoader(client_id=self.args.client_id, total_clients=self.args.total_clients)       
     


    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        
        # Load the data 
        (x_train, y_train) = self.data_loader.load_train_data(data_sampling_percentage=config["data_sample_percentage"])
       
        start_time = time.time()  # Capture the start time
       
        # Set the weights of the model
        model.set_weights(parameters)

        # config and the name of the container
        logger.info("Config for client with name: %s, is: %s", self.container_name, config)
        
        # Update the properties with the config as it contains the new configuration for the round
        self.properties.update(config)
    
        logger.info("Training model for client with container name: %s", self.container_name)
       
        # Train the model
        history = model.fit(x_train, y_train, batch_size=config["batch_size"], epochs=config["epochs"])
        

        # learning_rate_setter = LearningRateSetter(learning_rate= config["learning_rate"])
        # gradient_clipping_setter = GradientClippingCallback(clipvalue=config.get("gradient_clipping_value"))
        # model_precision_adjustment = ModelPrecisionAdjustmentCallback(target_dtype=config.get("model_precision"))

        # # Apply model adjustments based on config
        # model_adjuster = ModelAdjuster(model)
        # with model_adjuster.apply_adjustments(config):
        #     history = model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], callbacks=[learning_rate_setter, gradient_clipping_setter, model_precision_adjustment])      
    

        # Get the weights after training
        parameters_prime = model.get_weights()

        logger.info("Training completed for client with container name: %s", self.container_name)



        end_time = time.time()  # Capture the end time
        duration = end_time - start_time  # Calculate duration

        # Store operational metrics for fit
        self.fit_operational_metrics = {
            "fit_start_time": start_time,
            "fit_end_time": end_time,
            "fit_duration": duration
        }

        # Calculate evaluation metric
        results = {
            "accuracy": float(history.history["accuracy"][-1]),
            "start_time": start_time,
            "end_time": end_time,
            "container_name": self.container_name,
        }     
       
        # Check if sparsification is enabled
        if config["sparsification_enabled"]:
            
            logger.info("sparsification enabled for client with container name: %s", self.container_name)
            
            # Create a sparsifier object 
            sparsifier = Sparsifier(method=config["sparsification_method"], percentile=config["sparsification_percentile"])
            
            serialized_sparse_weights, total_nnz = sparsifier.sparsify_and_serialize_weights(parameters_prime)

            # Get the size of the serialized sparse weights and the original weights
            serialized_sparse_weights_size = calculate_weights_size(serialized_sparse_weights)
            original_weights_size = calculate_weights_size(parameters_prime)    
            
            logger.info("serialized_sparse_weights_size: %s", serialized_sparse_weights_size)
            logger.info("original_weights_size: %s", original_weights_size)
            
            # Return new weights, number of training examples, and results
            return serialized_sparse_weights, len(self.x_train), results

        else:
            # Directly return the dense weights without sparsification
            return parameters_prime, len(x_train), results
        

    def evaluate(self, parameters, config):
        
        (x_test, y_test) = self.data_loader.get_test_data()

        start_time = time.time()  # Capture the start time
        
        model.set_weights(parameters)

        # Evaluate the model and get the loss and accuracy
        loss, accuracy = model.evaluate(x_test, y_test)

        end_time = time.time()  # Capture the end time
        duration = end_time - start_time  # Calculate duration

        # Store operational metrics for evaluation
        self.eval_operational_metrics = {
            "eval_start_time": start_time,
            "eval_end_time": end_time,
            "eval_duration": duration,
            "test_set_size": len(x_test),
            "test_set_accuracy": accuracy
        }

        # Combine fit and eval operational metrics for logging
        combined_metrics = {**self.fit_operational_metrics, **self.eval_operational_metrics}

        # Log the combined operational metrics
        mlflow_helper = MlflowHelper()
        mlflow_helper.log_round_metrics_for_client(
            container_name=os.getenv('container_name'),
            server_round=config["server_round"],
            experiment_id=config["experiment_id"],
            operational_metrics=combined_metrics 
        )

        return float(loss), len(x_test), {"accuracy": float(accuracy)}
    

    def get_properties(self, *args, **kwargs):
            return self.properties




app = Flask(__name__)
client_api = Api(app)
CORS(app, resources={r"/client_api/*": {"origins": "*"}})

class StartFLClient(Resource):
    def get(self):
        try:
            client_thread = Thread(target=start_fl_client, daemon=True)
            client_thread.start()
            return {"message": "Started FL client"}, 200 
        except Exception as e:
            logger.error("Error starting FL client: %s", e)
            return {"status": "error", "message": str(e)}, 500

    
def start_fl_client():
    try:
        fl.client.start_numpy_client(server_address=args.server_address, client=Client())
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}

class Ping(Resource):
    def get(self):
        logger.info("Received ping. I'm client and I'm alive.")
        return 'I am client and I am alive', 200


client_api.add_resource(Ping, '/client_api/ping')
client_api.add_resource(StartFLClient, '/client_api/start-fl-client')

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT"))
    logger.info("Starting client on port %s", port)
    app.run(debug=True, threaded=True, host="0.0.0.0", port=port)
