import uuid
import mlflow
from strategy.strategy import FedCustom
import logging
import flwr as fl

logger = logging.getLogger(__name__)     

class Experiment:
    def __init__(
        self,
        rounds: int = 30,
        convergence_accuracy: float = 0.95,
    ) -> None:
        super().__init__()
        self.experiment_name = self.create_random_experiment_name()
        self.experiment_id = self.create_experiment()
        self.rounds = rounds
        self.convergence_accuracy = convergence_accuracy
        self.strategy = self.set_strategy()
        self.log_experiment_details()


    def set_strategy(self):
        """Configure the federated learning strategy for the experiment."""
        return FedCustom(experiment_id=self.experiment_id,convergence_accuracy=self.convergence_accuracy)

    @property
    def name(self):
        return str(self.experiment_name)

    def create_experiment(self):
        experiment_id = mlflow.create_experiment(self.name)
        return str(experiment_id)

    def log_experiment_details(self):
        """Logs general details about the experiment to MLflow."""
        with mlflow.start_run(experiment_id=self.experiment_id,run_name="Experiment Details"):
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("max_rounds", self.rounds)
            mlflow.log_param("convergence_accuracy", self.convergence_accuracy)
            # You can add other general details about the experiment here   
    
    def create_random_experiment_name(self):
        return str(uuid.uuid4())