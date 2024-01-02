import uuid
import mlflow
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

class MlflowHelper:
    def __init__(
        self,
        rounds: int = 100,
        convergence_accuracy: float = 0.8,
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.convergence_accuracy = convergence_accuracy

    @property
    def name(self):
        return str(self.experiment_name)

    def get_experiment_id(self):
        return str(self.experiment_id)

    def create_experiment(self):
        self.experiment_name = self.create_random_experiment_name()
        self.experiment_id = mlflow.create_experiment(self.name)
        logger.info(f"Created experiment {self.name} with id {self.experiment_id}")
        self.log_experiment_details()
        return str(self.experiment_id)

    def log_experiment_details(self):
        """Logs general details about the experiment to MLflow."""
        with mlflow.start_run(experiment_id=self.experiment_id,run_name="Experiment Details"):
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("max_rounds", self.rounds)
            mlflow.log_param("convergence_accuracy", self.convergence_accuracy)
            # You can add other general details about the experiment here   

    def create_random_experiment_name(self):
        return str(uuid.uuid4())


    def log_round_metrics_for_client(self,container_name,server_round,experiment_id,operational_metrics):

        with mlflow.start_run(experiment_id=experiment_id,run_name=f"{server_round} - round - {container_name}") as client_run:             
            mlflow.set_tag("container_name", container_name)
            mlflow.log_metric("eval_start_time", operational_metrics['eval_start_time'])
            mlflow.log_metric("eval_end_time", operational_metrics['eval_end_time'])
            mlflow.log_metric("eval_duration", operational_metrics['eval_duration'])
            mlflow.log_metric("test_set_size", operational_metrics['test_set_size'])
            mlflow.log_metric("test_set_accuracy", operational_metrics['test_set_accuracy'])
            mlflow.log_metric("fit_start_time", operational_metrics['fit_start_time'])
            mlflow.log_metric("fit_end_time", operational_metrics['fit_end_time'])
            mlflow.log_metric("fit_duration", operational_metrics['fit_duration'])

    
    def log_aggregated_metrics(self,experiment_id,server_round, loss_aggregated, accuracy_aggregated, results, failures):
        """Logs aggregated metrics for a server round to MLflow."""
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{server_round} - round - server"): 
            logger.info(f"Logging aggregated metrics for server round {server_round} for experiment_id {experiment_id}")
            
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



