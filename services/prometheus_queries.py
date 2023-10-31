
# Max cpu usage for a container between two timestamps
import logging
from utils.simple_utils import range_from_timestamps

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  


def container_specific_max_cpu_usage_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_cpu_usage_seconds_total') -> str:
    query = f'max_over_time({metric}{{name="{container_name}"}}[{start_timestamp}ms:{end_timestamp}ms])'
    return query

# Max memory usage for a container between two timestamps
def container_specific_max_memory_usage_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_memory_usage_bytes') -> str:
    query = f'max_over_time({metric}{{name="{container_name}"}}[{start_timestamp}ms:{end_timestamp}ms])'
    return query

# Average CPU usage for a container between two timestamps
def container_specific_rate_of_cpu_usase_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_cpu_usage_seconds_total') -> str:
    range_str = range_from_timestamps(start_timestamp)
    query = f'rate({metric}{{name="{container_name}"}}[{range_str}])'
    return query

# Average memory usage for a container between two timestamps
def container_specific_average_memory_usage_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_memory_usage_bytes') -> str:
    query = f'avg_over_time({metric}{{name="{container_name}"}}[{start_timestamp}ms:{end_timestamp}ms])'
    return query


# Max memory usage for a container between two timestamps
def container_specific_max_memory_usage_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_memory_usage_bytes') -> str:
    query = f'max_over_time({metric}{{name="{container_name}"}}[{start_timestamp}ms:{end_timestamp}ms])'
    return query

# Download bandwidth for a container between two timestamps in bits per second
def container_incoming_bandwidth_query(container_name: str, start_timestamp: int, end_timestamp: int, network_interface: str = 'eth0', metric: str = 'container_network_receive_bytes_total') -> str:
    range_str = range_from_timestamps(start_timestamp)
    incoming_bandwidth_query = f'8 * rate({metric}{{name="{container_name}", interface="{network_interface}"}}[{range_str}]) / 1000000'
    return incoming_bandwidth_query

# Upload bandwidth for a container between two timestamps in bits per second
def container_outgoing_bandwidth_query(container_name: str, start_timestamp: int, end_timestamp: int,network_interface: str = 'eth0', metric: str = 'container_network_transmit_bytes_total') -> str:
    range_str = range_from_timestamps(start_timestamp)
    outgoing_bandwidth_query = f'8 * rate({metric}{{name="{container_name}", interface="{network_interface}"}}[{range_str}]) / 1000000'
    return outgoing_bandwidth_query

# CPU allocation for a container
def container_cpu_allocation_query(container_name: str, metric_quota='container_spec_cpu_quota', metric_period='container_spec_cpu_period') -> str:
    query = f'({metric_quota}{{name="{container_name}"}} / {metric_period}{{name="{container_name}"}})'
    return query

# Container memory limit
def container_memory_limit_query(container_name: str, metric='container_spec_memory_limit_bytes') -> str:
    query = f'{metric}{{name="{container_name}"}}'
    return query

# Number of physical CPU cores
def machine_physical_cpu_cores_query(container_name: str,metric='machine_cpu_physical_cores') -> str:
    query = metric
    return query
