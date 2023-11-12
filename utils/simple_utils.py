import math
import time
import flwr as fl
import pickle
import numpy as np


def range_from_timestamps(start_timestamp: int) -> str:
    current_timestamp = int(time.time() * 1000)  # Current time in milliseconds
    range_seconds = (current_timestamp - start_timestamp) / 1000  # Convert milliseconds to seconds
    range_seconds_rounded = math.floor(range_seconds) if range_seconds > 0 else 1  # Round to the nearest integer, ensuring it's at least 1s
    range_str = f'{range_seconds_rounded}s'
    return range_str


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