from ast import Dict
import time
import os
import argparse
from flask import Flask, request, jsonify
from flask_cors.extension import CORS
from matplotlib import pyplot as plt
import requests
from flask_restful import Resource, Api, abort
import flwr as fl
from services.prometheus_queries import *
from services.prometheus_service import PrometheusService
import tensorflow as tf
import logging
from threading import Thread
import logging
import numpy as np
import logging
from helpers.load_data_helper import load_data_helper
import os
import numpy as np
import tensorflow as tf
from helpers.mlflow_helper import log_round_metrics_for_client
from models.cnn import cnn as cnn_model
from flwr.common import GetPropertiesIns,GetPropertiesRes,Status
from callbacks.learning_rate_setter import LearningRateSetter

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
parser = argparse.ArgumentParser(description='Flower client')
parser.add_argument('--server_address', type=str, default="server:8080",
                    help='Server address')
args = parser.parse_args()

model = cnn_model()

class Client(fl.client.NumPyClient):
    def __init__(self):
        # Initialize instance variables to store operational metrics
        self.fit_operational_metrics = {}
        self.eval_operational_metrics = {}
        self.properties = {}
        self.prom_service = PrometheusService()
        self.container_name = os.getenv('container_name')
        self.static_metrics = self.prom_service.get_container_static_metrics(self.container_name)
        self.properties.update({
            "container_name": self.container_name,
            **self.static_metrics
        })

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        start_time = time.time()  # Capture the start time
        # Set the weights of the model
        model.set_weights(parameters)

        logger.info("config for client is: %s", config)
        
        # Update the properties with the config as it contains the new configuration for the round
        self.properties.update(config)

        # Use the dataset API for training
        train_dataset, _, num_examples_train, _ = load_data_helper(percentage = config["data_sample_percentage"],batch_size=config["batch_size"])
        
        learning_rate_setter = LearningRateSetter(learning_rate= config["learning_rate"])
        
        # Train the model
        history = model.fit(train_dataset, epochs=config["epochs"], callbacks=[learning_rate_setter])
        
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
            "accuracy": float(history.history["accuracy"][config["epochs"]-1]),
        }         
        
        parameters_prime = model.get_weights()

        # Return new weights, number of training examples, and results
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        start_time = time.time()  # Capture the start time
        
        model.set_weights(parameters)

        # Use the dataset API for evaluation
        _, test_dataset, _, num_examples_test = load_data_helper()

        loss, accuracy = model.evaluate(test_dataset)
        
        end_time = time.time()  # Capture the end time
        duration = end_time - start_time  # Calculate duration

        # Store operational metrics for evaluation
        self.eval_operational_metrics = {
            "eval_start_time": start_time,
            "eval_end_time": end_time,
            "eval_duration": duration,
            "test_set_size": num_examples_test,
            "test_set_accuracy": accuracy
        }

        # Combine fit and eval operational metrics for logging
        combined_metrics = {**self.fit_operational_metrics, **self.eval_operational_metrics}

        # Log the combined operational metrics
        log_round_metrics_for_client(
            container_name=os.getenv('container_name'),
            server_round=config["server_round"],
            experiment_id=config["experiment_id"],
            operational_metrics=combined_metrics 
        )

        return float(loss), num_examples_test, {"accuracy": float(accuracy)}
    

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
