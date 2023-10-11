import os
import argparse
import pickle
import sys
from flask import Flask, request, jsonify
from flask_cors.extension import CORS
from matplotlib import pyplot as plt
import requests
from flask_restful import Resource, Api, abort
import flwr as fl
import tensorflow as tf
import logging
from threading import Thread
import logging
import docker
import mlflow
import numpy as np
import logging
from helpers.load_data_helper import load_data_helper
import os
import numpy as np
import tensorflow as tf
from helpers.mlflow_helper import log_metrics_for_client
logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
parser = argparse.ArgumentParser(description='Flower client')
parser.add_argument('--server_address', type=str, default="server:8080",
                    help='Server address')
args = parser.parse_args()
# Define the loss function
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
# Load model (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", loss_function, metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = load_data_helper()



class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        # Set the weights of the model
        model.set_weights(parameters)
        
        # Train the model for a specified number of epochs
        history = model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"])
        logger.info("Fit complete. History: %s", history.history)
        
        # Calculate evaluation metric
        results = {
            "accuracy": float(history.history["accuracy"][config["epochs"]-1]),
        }         
        
        parameters_prime = model.get_weights()


        num_examples_train = len(x_train)

        # Return new weights, number of training examples, and results
        return parameters_prime, num_examples_train, results


    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        # log metrics to MLflow for this client
        log_metrics_for_client(container_name=os.getenv('container_name'),server_round = config["server_round"],experiment_id=config["experiment_id"], model_performance={'test_accuracy': accuracy})

        return float(loss), len(x_test), {"accuracy": float(accuracy)}


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

class Ping(Resource):
    def get(self):
        logger.info("Received ping. I'm client and I'm alive.")
        return 'I am client and I am alive', 200


class ContainerMetrics(Resource):
    def get(self):
        container_name = os.getenv('container_name')
        if container_name is None:
            logger.error("container_name environment variable is not set.")
        else:
            stats = log_metrics_for_client(container_name)

        return {"cpu_stats": stats['cpu_stats'], "memory_stats": stats['memory_stats']}, 200
     
         
    
def start_fl_client():
    try:
        fl.client.start_numpy_client(server_address=args.server_address, client=Client())
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


client_api.add_resource(Ping, '/client_api/ping')
client_api.add_resource(StartFLClient, '/client_api/start-fl-client')
client_api.add_resource(ContainerMetrics, '/client_api/container-metrics')

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT"))
    logger.info("Starting client on port %s", port)
    app.run(debug=True, threaded=True, host="0.0.0.0", port=port)
