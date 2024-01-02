import argparse
import os
import traceback
import flwr as fl
import requests
from flask import Flask
from flask_cors.extension import CORS
from flask_restful import Resource, Api
import logging
from helpers.mlflow import MlflowHelper
from strategy.strategy import FedCustom
from utils.simple_utils import parse_docker_compose
from prometheus_client import start_http_server, Gauge
import threading


# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a gauge to track the global model accuracy
accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the global model')
# Define a gauge to track the global model loss
loss_gauge = Gauge('model_loss', 'Current loss of the global model')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flower Server')
parser.add_argument('--number_of_rounds',type=int, default=100, help="Number of FL rounds (default: 100)")
parser.add_argument('--convergence_accuracy', type=float, default=0.8, help='Convergence accuracy (default: 0.8)')


args = parser.parse_args()


# Flask App and API Configuration
app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Function to Start Federated Learning Server
def start_fl_server(strategy):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=args.number_of_rounds),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"FL Server error: {e}", exc_info=True)


def start_prometheus_server(port):
        try:
            start_http_server(port)
        except OSError as e:
            print(f"Unable to start Prometheus server: {e}")


# Resource to Handle Starting of FL Process
class StartFL(Resource):
    def get(self):

        # Initialize MLflow Experiment Instance
        experiment = MlflowHelper(convergence_accuracy = args.convergence_accuracy, rounds = args.number_of_rounds)
        
        # Initialize Strategy Instance
        strategy_instance = FedCustom(experiment_id=experiment.get_experiment_id(), accuracy_gauge=accuracy_gauge, loss_gauge=loss_gauge, convergence_accuracy = args.convergence_accuracy)
        
        # Start FL Server
        server_thread = threading.Thread(target=start_fl_server, args=(strategy_instance,), daemon=True)
        server_thread.start()
        server_thread.join(timeout=5)

        clients = parse_docker_compose("docker-compose.yml")
        for client in clients:
            try:
                response = requests.get(f"http://{client}/client_api/start-fl-client")
                logger.info(f"Response from client {client}: {response.text}")
            except Exception as e:
                logger.error(f"Error with client {client}: {e}", exc_info=True)
                return {"status": "error", "message": str(e), "trace": traceback.format_exc()}, 500

        return {"status": "started", "experiment_name": experiment.name}, 200


# Resource for Health Check
class Ping(Resource):
    def get(self):
        logger.info("Ping received.")
        return 'Server is alive', 200


# Add Resources to API
api.add_resource(Ping, '/api/ping')
api.add_resource(StartFL, '/api/start-fed-learning')

# Main Function
if __name__ == "__main__":

    # Start Prometheus Metrics Server on a separate thread
    threading.Thread(target=start_prometheus_server, args=(8000,), daemon=True).start()

    port = int(os.environ.get("FLASK_RUN_PORT", 6000))
    app.run(debug=True, threaded=True, host="0.0.0.0", port=port)