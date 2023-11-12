import datetime
import pickle
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import flwr as fl
import logging
import mlflow
import sys
import numpy as np
from strategy.client_selector import ClientSelector
from utils.load_configs import load_strategy_config
import sparse 
from flwr.common import GetPropertiesIns
from utils.simple_utils import get_client_properties

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

    
class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        experiment_id: str = None,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        converged: bool = False,
        convergence_accuracy: float = None,
        round_timestamps: Dict[int, Dict[str, datetime.datetime]] = {},
    ) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.converged = converged
        self.convergence_accuracy = convergence_accuracy
        self.round_timestamps = round_timestamps


        logger.info("FedCustom strategy initialized.")
        logger.info("Configuring evaluation with experiment ID: %s", self.experiment_id)


    def __repr__(self) -> str:
        return "FedCustom"

    def check_convergence(self, accuracy: float) -> None:
        if accuracy >= self.convergence_accuracy:
            self.converged = True

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        logger.info("Initializing parameters.")
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        logger.info(f"Configuring fit for server round {server_round}.")

        # Initialize the round timestamps
        self.round_timestamps[server_round] = {"start": int(time.time() * 1000)}
        
        selected_clients = []

        # Check if this is not the first round and the previous round timestamps are available
        if server_round > 0 and server_round - 1 in self.round_timestamps:

            client_selector = ClientSelector(client_manager)  
            all_clients = client_selector.get_all_clients()
            selected_clients = client_selector.filter_clients_by_criteria(all_clients, server_round, self.round_timestamps)
            logger.info(f"Selected clients based on criteria are: {selected_clients}")

       
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = selected_clients if selected_clients else client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        standard_config = load_strategy_config()
        logger.info(f"Standard config is: {standard_config}")

        if selected_clients:
            # Use the standard config as a default and update it with the client-specific config if available
            fit_configurations = [
                (client_info['client'], FitIns(parameters, {**standard_config, **client_info.get('config', {})}))
                for client_info in clients
            ]
        else:
            # Use the standard config for all clients
            fit_configurations = [
                (client, FitIns(parameters, standard_config))
                for idx, client in enumerate(clients)
            ]

    
        return fit_configurations


    def deserialize_sparse_coo(self,serialized_data):
        # Deserialize the data
        deserialized_data = pickle.loads(serialized_data)

        # Extract components
        data = np.array(deserialized_data["data"])
        coords = np.array(deserialized_data["coords"])
        shape = tuple(deserialized_data["shape"])

        # Reconstruct the sparse COO matrix
        return sparse.COO(coords, data, shape=shape)


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and convert from serialized sparse to dense format."""
        logger.info(f"Aggregating fit results for server round {server_round}.")

        # Convert from serialized sparse to dense format and prepare for aggregation
        dense_results = []
        for ClientProxy, fit_res in results:
            # Get the client properties
            client_properties = get_client_properties(ClientProxy)
            enabledSparsification =  client_properties.get('sparsification')

            if enabledSparsification:
                # Deserialize each sparse weight and convert to dense
                dense_weights = []
                for serialized_sparse_weight in parameters_to_ndarrays(fit_res.parameters):
                    # Deserialize sparse weight (this is where you add your deserialization logic)
                    sparse_weight = self.deserialize_sparse_coo(serialized_sparse_weight)
                    
                    # Convert to dense format
                    dense_weight = sparse_weight.todense()
                    dense_weights.append(dense_weight)

                dense_results.append((dense_weights, fit_res.num_examples))

            else:
                dense_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

        # Use Flower's default aggregation method on dense weights
        aggregated_weights = aggregate(dense_results)

        # Convert aggregated weights back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated



    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {"server_round": server_round,"experiment_id":self.experiment_id}

        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracy using weighted average."""
        
        if not results:
            return None, {}

        logger.info(f"Aggregating evaluation results for server round {server_round} are {results}.")
        
        # Calculate weighted average for loss using the provided function
        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )

        # Calculate weighted average for accuracy
        accuracies = [evaluate_res.metrics["accuracy"] * evaluate_res.num_examples for _, evaluate_res in results]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy_aggregated = sum(accuracies) / sum(examples) if sum(examples) != 0 else 0

        
        # After aggregating the accuracy:
        self.check_convergence(accuracy_aggregated)
        
        # If converged, log the information, perform cleanup and raise an exception to stop the server
        if self.converged:
            self.handle_convergence(accuracy_aggregated)
            raise Exception("Convergence criteria met. Stopping server.")

        metrics_aggregated = {
            "loss": loss_aggregated, 
            "accuracy": accuracy_aggregated
        }

        # Log the aggregated loss and accuracy to MLflow
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"{server_round} - round - server"): 
            mlflow.log_metric("aggregated_loss", loss_aggregated)
            mlflow.log_metric("aggregated_accuracy", accuracy_aggregated)
            
            # log the failure rate
            denominator = len(results) + len(failures)
            failure_rate = len(failures) / denominator if denominator != 0 else 0
            mlflow.log_metric("failure_rate", failure_rate)

            mlflow.set_tag("server_round", server_round)
            mlflow.set_tag("container_name", "fl-server")

            # log the number of clients that participated in this round
            mlflow.log_metric("num_clients", len(results))
        
        self.round_timestamps[server_round]["end"] = int(time.time() * 1000)

        return loss_aggregated, metrics_aggregated



    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, FitRes]],
    #     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     """Aggregate fit results using weighted average."""
    #     logger.info(f"Aggregating fit results for server round {server_round}.")
    #     weights_results = [
    #         (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
    #         for _, fit_res in results
    #     ]
    #     parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
    #     metrics_aggregated = {}
    #     return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        logger.info(f"Evaluating model for server round {server_round}.")
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        logger.info(f"Determining number of fit clients from {num_available_clients} available clients.")
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        logger.info(f"Determining number of evaluation clients from {num_available_clients} available clients.")
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def handle_convergence(self, accuracy_aggregated):
        # Log the convergence
        logger.info(f"Convergence achieved with accuracy: {accuracy_aggregated:.2f}%")

        
