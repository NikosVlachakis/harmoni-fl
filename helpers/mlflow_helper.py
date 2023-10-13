import os
import docker
import requests
import mlflow
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

def get_container_id(container_name):
    client = docker.from_env()
    container = client.containers.get(container_name)
    return container.id

def get_container_stats(container_id):
    # Get Docker host IP from environment variable
    docker_host_ip = os.getenv('DOCKER_HOST_IP')
    if docker_host_ip is None:
        raise ValueError('The DOCKER_HOST_IP environment variable is not set.')
    url = f"http://{docker_host_ip}:2375/containers/{container_id}/stats?stream=false"
    response = requests.get(url)
    if response.status_code == 200:
        stats = response.json()
        return stats
    else:
        return None

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




def log_container_metrics_for_client(container_name,server_round,experiment_id, model_performance=None):
    
    container_id = get_container_id(container_name)
    stats = get_container_stats(container_id)


    if stats:
        with mlflow.start_run(experiment_id=experiment_id,run_name=f"{server_round} - round - {container_name}") as client_run:
            
            cpu_stats = stats['cpu_stats']
            mem_stats = stats['memory_stats']

            # log the container's name as a tag
            mlflow.set_tag("container_name", container_name)
            
            # memory usage
            memory_limit = mem_stats['limit']
            mlflow.log_metric("memory_limit", memory_limit/1000000)
        
            mlflow.log_metric("online_cpus", cpu_stats['online_cpus'])

            # log model performance
            if model_performance is not None:
                mlflow.log_metric("test_accuracy", model_performance['test_accuracy'])

        
        return stats
    else:
        return None
