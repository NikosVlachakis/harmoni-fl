import uuid
import mlflow

class MlflowHelper:
    def __init__(
        self,
        rounds: int = 100,
        convergence_accuracy: float = 0.8,
    ) -> None:
        super().__init__()
        self.experiment_name = self.create_random_experiment_name()
        self.experiment_id = self.create_experiment()
        self.rounds = rounds
        self.convergence_accuracy = convergence_accuracy
        self.log_experiment_details()

    @property
    def name(self):
        return str(self.experiment_name)

    def get_experiment_id(self):
        return str(self.experiment_id)

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


    def log_round_metrics_for_client(container_name,server_round,experiment_id,operational_metrics):
    
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




