# metric_names.py

from enum import Enum, auto

class Names(Enum):
    MAX_CPU_USAGE = "max_cpu_usage"
    MAX_MEMORY_USAGE_PERCENTAGE = "max_memory_usage_percentage"
    INCOMING_BANDWIDTH = "incoming_bandwidth"
    OUTGOING_BANDWIDTH = "outgoing_bandwidth"
    LEARNING_RATE_BASED_ON_INCOMING_BANDWIDTH = "learning_rate_based_on_incoming_bandwidth"
    EPOCH_ADJUSTMENT_BASED_ON_CPU_UTILIZATION = "epoch_adjustment_based_on_cpu_utilization"
    CONTAINER_CPU_CORES = "container_cpu_cores"
    CONTAINER_MEMORY_LIMIT = "container_memory_limit"

