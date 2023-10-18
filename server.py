import os
import uuid
import traceback
import flwr as fl
import requests
from flask import Flask, request, jsonify
from flask_cors.extension import CORS
from flask_restx import Resource, Api
import logging
from threading import Thread
from experiment.experiment import Experiment

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

app = Flask(__name__)
api = Api(app, version="1.0", title="Federated Learning Server", description="API for the Federated Learning Server with Swagger")
CORS(app, resources={r"/api/*": {"origins": "*"}})

clients = os.environ['CLIENTS'].split(',')
logger.info("Clients: %s", clients)

class StartFL(Resource):
    def get(self):
        job_id = str(uuid.uuid4())  # Generate a unique job ID

        # Create and configure a new experiment
        experiment = Experiment()
        
        server_thread = Thread(target=start_fl_server,args=(job_id, experiment.strategy, experiment.rounds), daemon=True)
        server_thread.start()
        
        # Wait for the server to start up

        server_thread.join(timeout=5)

        # Send start signal to clients
        for client in clients:
            try:
                r = requests.get(f"http://{client}/client_api/start-fl-client")
                logger.info("Response from client %s: %s", client, r.text)
            except Exception as e:
                exception_traceback = traceback.format_exc()
                print(f"Error sending start signal to client {client}: {e}\n{exception_traceback}")
                return {"status": "error", "message": str(e), "trace": exception_traceback}, 500

        response = {"status": "started", "job_id": job_id, "experiment_name": experiment.name}

        return response, 200


def start_fl_server(job_id, strategy,rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )        
        
    except Exception as e:
        logger.info(f"Server stopped with message: {str(e)}")


class Ping(Resource):
    def get(self):
        logger.info("Received ping. I'm server and I'm alive.")
        return 'I am server and I am alive', 200
    

api.add_resource(Ping, '/api/ping')
api.add_resource(StartFL, '/api/start-fed-learning')


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT"))
    app.run(debug=True, threaded=True, host="0.0.0.0", port=port)















# import os
# import math
# import argparse
# import time
# from typing import Dict, List, Tuple

# import tensorflow as tf

# import flwr as fl
# from flwr.common import Metrics
# import logging

# logging.basicConfig(level=logging.INFO)  # Configure logging
# logger = logging.getLogger(__name__)     # Create logger for the module

# from strategies.custom_strategy import FedCustom
# from experiment.experiment import Experiment
# import ray

# ray.init(dashboard_host='0.0.0.0')


# # Make TensorFlow logs less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")

# parser.add_argument(
#     "--num_cpus",
#     type=int,
#     default=1,
#     help="Number of CPUs to assign to a virtual client",
# )
# parser.add_argument(
#     "--num_gpus",
#     type=float,
#     default=0.0,
#     help="Ratio of GPU memory to assign to a virtual client",
# )
# parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds.")

# NUM_CLIENTS = 100
# VERBOSE = 0


# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, x_train, y_train, x_val, y_val) -> None:
#         # Create model
#         self.model = get_model()
#         self.x_train, self.y_train = x_train, y_train
#         self.x_val, self.y_val = x_val, y_val

#     def get_parameters(self, config):
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         self.model.fit(
#             self.x_train, self.y_train, epochs=1, batch_size=32, verbose=VERBOSE
#         )
#         return self.model.get_weights(), len(self.x_train), {}

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         loss, acc = self.model.evaluate(
#             self.x_val, self.y_val, batch_size=64, verbose=VERBOSE
#         )
#         return loss, len(self.x_val), {"accuracy": acc}


# def get_model():
#     """Constructs a simple model architecture suitable for MNIST."""
#     model = tf.keras.models.Sequential(
#         [
#             tf.keras.layers.Flatten(input_shape=(28, 28)),
#             tf.keras.layers.Dense(128, activation="relu"),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(10, activation="softmax"),
#         ]
#     )
#     model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
#     return model


# def get_client_fn(dataset_partitions):
#     """Return a function to construc a client.

#     The VirtualClientEngine will exectue this function whenever a client is sampled by
#     the strategy to participate.
#     """

#     def client_fn(cid: str) -> fl.client.Client:
#         """Construct a FlowerClient with its own dataset partition."""

#         # Extract partition for client with id = cid
#         x_train, y_train = dataset_partitions[int(cid)]
#         # Use 10% of the client's training data for validation
#         split_idx = math.floor(len(x_train) * 0.9)
#         x_train_cid, y_train_cid = (
#             x_train[:split_idx],
#             y_train[:split_idx],
#         )
#         x_val_cid, y_val_cid = x_train[split_idx:], y_train[split_idx:]

#         # Create and return client
#         return FlowerClient(x_train_cid, y_train_cid, x_val_cid, y_val_cid)

#     return client_fn


# def partition_mnist():
#     """Download and partitions the MNIST dataset."""
#     (x_train, y_train), testset = tf.keras.datasets.mnist.load_data()
#     partitions = []
#     # We keep all partitions equal-sized in this example
#     partition_size = math.floor(len(x_train) / NUM_CLIENTS)
#     for cid in range(NUM_CLIENTS):
#         # Split dataset into non-overlapping NUM_CLIENT partitions
#         idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
#         partitions.append((x_train[idx_from:idx_to] / 255.0, y_train[idx_from:idx_to]))
#     return partitions, testset


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     """Aggregation function for (federated) evaluation metrics.

#     It ill aggregate those metrics returned by the client's evaluate() method.
#     """
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}


# def get_evaluate_fn(testset):
#     """Return an evaluation function for server-side (i.e. centralised) evaluation."""
#     x_test, y_test = testset

#     # The `evaluate` function will be called after every round by the strategy
#     def evaluate(
#         server_round: int,
#         parameters: fl.common.NDArrays,
#         config: Dict[str, fl.common.Scalar],
#     ):
#         model = get_model()  # Construct the model
#         model.set_weights(parameters)  # Update model with the latest parameters
#         loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
#         return loss, {"accuracy": accuracy}

#     return evaluate


# def main() -> None:
#     # Parse input arguments
#     args = parser.parse_args()

#     # Create dataset partitions (needed if your dataset is not pre-partitioned)
#     partitions, testset = partition_mnist()

#     # Create FedAvg strategy
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=0.1,  # Sample 10% of available clients for training
#         fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
#         min_fit_clients=4,  # Never sample less than 10 clients for training
#         min_evaluate_clients=4,  # Never sample less than 5 clients for evaluation
#         min_available_clients=int(
#            6
#         ),  # Wait until at least 75 clients are available
#         evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
#         evaluate_fn=get_evaluate_fn(testset),  # global evaluation function
#     )

#     # With a dictionary, you tell Flower's VirtualClientEngine that each
#     # client needs exclusive access to these many resources in order to run
#     client_resources = {
#         "num_cpus": args.num_cpus,
#         "num_gpus": args.num_gpus,
#     }
#     experiment = Experiment()

#     # Start simulation
#     fl.simulation.start_simulation(
#         client_fn=get_client_fn(partitions),
#         num_clients=15,
#         config=fl.server.ServerConfig(num_rounds=10),
#         strategy=experiment.strategy,
#         # strategy = strategy,
#         ray_init_args = {"include_dashboard": True,'_temp_dir': '/SharedVolume'},
#     )


# if __name__ == "__main__":
#     main()

   