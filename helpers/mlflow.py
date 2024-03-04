import json
import os
import uuid
import mlflow
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

class MlflowHelper:
    def __init__(
        self,
        rounds: int = 100,
        convergence_accuracy: float = 0.8,
        client_id: int = 1,
        dp_opt: int = 0
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.convergence_accuracy = convergence_accuracy
        self.client_id = client_id
        self.dp_opt = dp_opt == 1
        self.epsilon_values_file_path = f"mlflow/epsilon_data_client{self.client_id}.csv"

    @property
    def name(self):
        return 'v2' + str(self.experiment_name)

    def get_experiment_id(self):
        return str(self.experiment_id)

    def create_experiment(self):
        self.experiment_name = self.create_random_experiment_name()
        self.experiment_name = self.experiment_name
        self.experiment_id = mlflow.create_experiment(name=self.name)
        logger.info(f"Created experiment {self.name} with id {self.experiment_id}")
        self.log_experiment_details()
        return str(self.experiment_id)

    def log_experiment_details(self):
        """Logs general details about the experiment to MLflow."""
        
        with mlflow.start_run(experiment_id=self.experiment_id,run_name="Experiment Details"):
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("max_rounds", self.rounds)
            mlflow.log_param("convergence_accuracy", self.convergence_accuracy)
            mlflow.log_param("experiment_description", "tool-enabled")
            # mlflow.log_param("experiment_description", "without-tool")


            # You can add other general details about the experiment here   

    def create_random_experiment_name(self):
        return str(uuid.uuid4())


    def log_round_metrics_for_client(self,container_name,server_round,experiment_id,operational_metrics, properties):
        # Define total epochs and iterations
        
        # Start a new MLflow run
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{server_round} - round - {container_name}") as client_run:
            mlflow.set_tag("container_name", container_name)
            mlflow.set_tag("server_round", server_round)
            mlflow.log_params(properties)
            mlflow.log_metrics(operational_metrics)
            if self.dp_opt:
                mlflow.log_artifact(local_path= self.epsilon_values_file_path)

            
            

    
    def log_aggregated_metrics(self,experiment_id,server_round, loss_aggregated, accuracy_aggregated, results, dropped_out):
        """Logs aggregated metrics for a server round to MLflow."""
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{server_round} - round - server"): 
            
            mlflow.log_metric("aggregated_loss", loss_aggregated)
            mlflow.log_metric("aggregated_accuracy", accuracy_aggregated)

            # log the dropped_out rate
            mlflow.log_metric("dropped_out", dropped_out)

            mlflow.set_tag("server_round", server_round)
            mlflow.set_tag("container_name", "fl-server")

            # log the number of clients that participated in this round
            mlflow.log_metric("num_clients", len(results))



