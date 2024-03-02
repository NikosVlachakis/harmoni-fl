import argparse
import flwr as fl
import logging
from helpers.mlflow import MlflowHelper
from strategy.strategy import FedCustom
from prometheus_client import start_http_server, Gauge

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flower Server')
parser.add_argument('--number_of_rounds',type=int, default=100, help="Number of FL rounds (default: 100)")
parser.add_argument('--convergence_accuracy', type=float, default=0.8, help='Convergence accuracy (default: 0.8)')

args = parser.parse_args()

# Define a gauge to track the global model accuracy
accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the global model')
# Define a gauge to track the global model loss
loss_gauge = Gauge('model_loss', 'Current loss of the global model')

# Function to Start Federated Learning Server
def start_fl_server(strategy):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            # config=fl.server.ServerConfig(num_rounds=args.number_of_rounds),
            config=fl.server.ServerConfig(num_rounds=150),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"FL Server error: {e}", exc_info=True)


def start_prometheus_server(port):
        try:
            start_http_server(port)
        except OSError as e:
            print(f"Unable to start Prometheus server: {e}")



if __name__ == "__main__":

    # Start Prometheus Metrics Server
    start_prometheus_server(port=8000)

    # Initialize MLflow Experiment Instance
    mlflow_experiment = MlflowHelper(convergence_accuracy = args.convergence_accuracy, rounds = args.number_of_rounds)
    
    # Create a new experiment
    mlflow_experiment.create_experiment()
    
    # Initialize Strategy Instance
    strategy_instance = FedCustom(experiment_id=mlflow_experiment.get_experiment_id(), accuracy_gauge=accuracy_gauge, loss_gauge=loss_gauge, convergence_accuracy = args.convergence_accuracy)
    
    # Start FL Server
    start_fl_server(strategy_instance)