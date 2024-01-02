import math

def container_cpu_usage_percentage(container_name: str, start_timestamp: int, end_timestamp: int, cpu_usage_metric: str = 'container_cpu_usage_seconds_total', cpu_quota_metric: str = 'container_spec_cpu_quota', cpu_period_metric: str = 'container_spec_cpu_period') -> str:
    duration = math.ceil(end_timestamp - start_timestamp)
    query = (
            f"sum(rate({cpu_usage_metric}{{name=\"{container_name}\"}}[{duration}s] @{end_timestamp} )) / "
            f"sum({cpu_quota_metric}{{name=\"{container_name}\"}} / {cpu_period_metric}{{name=\"{container_name}\"}}) * 100"
        )
    return query


def container_memory_usage_percentage(container_name: str, start_timestamp: int, end_timestamp: int, memory_usage_metric: str = 'container_memory_working_set_bytes', memory_limit_metric: str = 'container_spec_memory_limit_bytes') -> str:
    duration = math.ceil(end_timestamp - start_timestamp)
    query = (
        f"avg_over_time({memory_usage_metric}{{name=\"{container_name}\"}}[{duration}s] @ {end_timestamp}) / "
        f"avg_over_time({memory_limit_metric}{{name=\"{container_name}\"}}[{duration}s] @ {end_timestamp}) * 100"
    )
    return query

def container_incoming_bandwidth_query(container_name: str, start_timestamp: int, end_timestamp: int, network_interface: str = 'eth0', metric: str = 'container_network_receive_bytes_total') -> str:
    duration = math.ceil(end_timestamp - start_timestamp)
    incoming_bandwidth_query = (
        f"rate({metric}{{name=\"{container_name}\", interface=\"{network_interface}\"}}[{duration}s] @ {end_timestamp}) / 1024^2"
    )
    return incoming_bandwidth_query

def container_outgoing_bandwidth_query(container_name: str, start_timestamp: int, end_timestamp: int, network_interface: str = 'eth0', metric: str = 'container_network_transmit_bytes_total') -> str:
    duration = math.ceil(end_timestamp - start_timestamp)
    outgoing_bandwidth_query = (
        f"rate({metric}{{name=\"{container_name}\", interface=\"{network_interface}\"}}[{duration}s] @ {end_timestamp}) / 1024^2"
    )
    return outgoing_bandwidth_query
