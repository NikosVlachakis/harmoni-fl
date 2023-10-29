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
    ADAPTIVE_BATCH_SIZE_BASED_ON_MEMORY_UTILIZATION = "adaptive_batch_size_based_on_memory_utilization"
    ADAPTIVE_DATA_SAMPLING_BASED_ON_MEMORY_UTILIZATION = "adaptive_data_sampling_based_on_memory_utilization"
    MODEL_LAYER_FREEZING_BASED_ON_HIGH_CPU_UTILIZATION = "model_layer_freezing_based_on_high_cpu_utilization"
    FREEZE_LAYERS_PERCENTAGE = "freeze_layers_percentage"
    GRADIENT_CLIPPING_BASED_ON_HIGH_CPU_UTILIZATION = "gradient_clipping_based_on_high_cpu_utilization"
    WEIGHT_PRECISION_BASED_ON_HIGH_CPU_UTILIZATION = "weight_precision_based_on_high_cpu_utilization"


