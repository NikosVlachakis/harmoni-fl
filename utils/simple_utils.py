import math
import os
import re
import time
import flwr as fl
import pickle
import numpy as np
import yaml



def get_client_properties(client_proxy, config={}, timeout=30):
    """
    Retrieves properties from a client.

    Args:
        client_proxy (ClientProxy): The proxy object for the client.
        config (dict, optional): Configuration parameters to be sent to the client. Defaults to {}.
        timeout (int, optional): Timeout for the request in seconds. Defaults to 30.

    Returns:
        dict: A dictionary containing the client properties.
    """
    # Request properties from the client
    properties_response = client_proxy.get_properties(fl.common.GetPropertiesIns(config=config), timeout=timeout)
    
    # Extract and return the properties
    client_properties = properties_response.properties
    return client_properties



def calculate_weights_size(weights):
    weights_size = sum(len(pickle.dumps(weight)) for weight in weights)
    return weights_size



def flatten_weights(weights):
    flat_weights = np.concatenate([
        np.array(weight).flatten() if not isinstance(weight, np.ndarray) else weight.flatten()
        for weight in weights
    ])
    return flat_weights


def parse_docker_compose(file_path):
    with open(file_path, 'r') as file:
        docker_compose = yaml.safe_load(file)

    client_services = []
    for service_name, service_details in docker_compose['services'].items():
        if re.match(r'^client\d+$', service_name):
            port = service_details.get('ports', [''])[0].split(':')[0]  # Extract the exposed port
            client_services.append(f"{service_name}:{port}")

    return client_services


def save_data_to_csv(path, data):
        data.to_csv(path, mode='w', header=True, index=False)
    