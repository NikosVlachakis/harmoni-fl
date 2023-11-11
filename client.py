from ast import Dict
import time
import os
import argparse
from flask import Flask, request, jsonify
from flask_cors.extension import CORS
from matplotlib import pyplot as plt
import requests
from models.callbacks.gradient_clipping_setter import GradientClippingCallback
from models.callbacks.model_precision_setter import ModelPrecisionAdjustmentCallback
from flask_restful import Resource, Api
import flwr as fl
from models.model_adjustments import ModelAdjuster
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
from helpers.mlflow_helper import log_round_metrics_for_client
from models.cnn import cnn as cnn_model
from models.callbacks.learning_rate_setter import LearningRateSetter
import sparse
import pickle

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
parser = argparse.ArgumentParser(description='Flower client')
parser.add_argument('--server_address', type=str, default="server:8080",
                    help='Server address')
args = parser.parse_args()

model = cnn_model()


def serialize_sparse_coo(coo_matrix):
    data = coo_matrix.data.tolist()  # Convert data to list
    coords = coo_matrix.coords.tolist()  # Convert coords to list
    shape = list(coo_matrix.shape)  # Convert shape to list

    # Create a dictionary with all components
    serialized_data = {
        "data": data,
        "coords": coords,
        "shape": shape
    }

    # Serialize the dictionary using pickle or a similar library
    return pickle.dumps(serialized_data)

def sparsify_weight(weight, sparsity_threshold=0.6):
        """Convert a weight to sparse format using a sparsity threshold."""
        mask = np.abs(weight) > sparsity_threshold
        weight_masked = np.where(mask, weight, 0)  # Apply mask
        return sparse.COO.from_numpy(weight_masked)  # Convert to sparse COO format

class Client(fl.client.NumPyClient):
    def __init__(self):
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

        enable_sparsification = True if self.container_name == "client1" else False
       
        if enable_sparsification:
            config["sparsification"] = True
        else:
            config["sparsification"] = False

        # Update the properties with the config as it contains the new configuration for the round
        self.properties.update(config)

        # Use the dataset API for training
        train_dataset, _, num_examples_train, _ = load_data_helper(percentage = config["data_sample_percentage"],batch_size=config["batch_size"])
        
        learning_rate_setter = LearningRateSetter(learning_rate= config["learning_rate"])
        gradient_clipping_setter = GradientClippingCallback(clipvalue=config.get("gradient_clipping_value"))
        model_precision_adjustment = ModelPrecisionAdjustmentCallback(target_dtype=config.get("model_precision"))

        # Apply model adjustments based on config
        model_adjuster = ModelAdjuster(model)
        with model_adjuster.apply_adjustments(config):
            history = model.fit(train_dataset, epochs=config["epochs"], callbacks=[learning_rate_setter, gradient_clipping_setter,model_precision_adjustment])

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
        

        if enable_sparsification:
            logger.info("sparsification enabled for client with container name: %s", self.container_name)

            # Sparsify each weight and serialize
            # !!!
            # serialized_sparse_weights = [serialize_sparse_coo(sparsify_weight(weight)) for weight in parameters_prime]
            # !!!
           
           
            
            # //////////
            
            # Sparsify each weight and serialize
            total_nnz = 0  # Initialize total non-zero element count
            serialized_sparse_weights = []
            for weight in parameters_prime:
                sparse_weight = sparsify_weight(weight)
                serialized_sparse_weight = serialize_sparse_coo(sparse_weight)
                serialized_sparse_weights.append(serialized_sparse_weight)

                # Log nnz for the current sparse weight and accumulate total nnz
                current_nnz = sparse_weight.nnz
                total_nnz += current_nnz

            # Log the total nnz for all weights
            logger.info("Total nnz for all sparse weights: %s", total_nnz)

            # /////////
            
            # get the size of the serialized sparse weights
            serialized_sparse_weights_size = sum(len(pickle.dumps(weight)) for weight in serialized_sparse_weights)
            logger.info("serialized_sparse_weights_size: %s", serialized_sparse_weights_size)
            # get the size of the parameters_prime
            original_weights_size = sum(len(pickle.dumps(weight)) for weight in parameters_prime)
            logger.info("original_weights_size: %s", original_weights_size)
            
            # Return new weights, number of training examples, and results
            return serialized_sparse_weights, num_examples_train, results

        else:
            # Directly return the dense weights without sparsification
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
