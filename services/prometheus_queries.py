from utils.metric_names import MetricNames


# Average cpu usage for a container_specific_avg_cpu_usage_query between two timestamps
def container_specific_avg_cpu_usage_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_cpu_usage_seconds_total') -> str:
    query = f'avg_over_time({metric}{{name="{container_name}"}}[{start_timestamp}ms:{end_timestamp}ms])'
    return query

# Max cpu usage for a container_specific_avg_cpu_usage_query between two timestamps
def container_specific_max_cpu_usage_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_cpu_usage_seconds_total') -> str:
    query = f'max_over_time({metric}{{name="{container_name}"}}[{start_timestamp}ms:{end_timestamp}ms])'
    return query

# Max memory usage for a container_specific_avg_cpu_usage_query between two timestamps
def container_specific_max_memory_usage_query(container_name: str, start_timestamp: int, end_timestamp: int, metric: str = 'container_memory_usage_bytes') -> str:
    query = f'max_over_time({metric}{{name="{container_name}"}}[{start_timestamp}ms:{end_timestamp}ms])'
    return query

METRIC_TO_QUERY_MAPPING = {
    MetricNames.AVG_CPU_USAGE.value: container_specific_avg_cpu_usage_query,
    MetricNames.MAX_CPU_USAGE.value: container_specific_max_cpu_usage_query,
    MetricNames.MAX_MEMORY_USAGE.value: container_specific_max_memory_usage_query
}