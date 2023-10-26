from criteria import *
from utils.names import Names
from services.prometheus_queries import *

CRITERIA_CONFIG = {
    Names.MAX_CPU_USAGE.value: {
        "query_func": container_specific_cpu_usage_query,
        "criterion_class": MaxCPUUsageCriterion
    },
    Names.MAX_MEMORY_USAGE_PERCENTAGE.value: {
        "query_func": container_specific_max_memory_usage_query,
        "criterion_class": MAXMemoryUsageCriterion
    },
    Names.LEARNING_RATE_BASED_ON_INCOMING_BANDWIDTH.value: {
        "query_func": container_incoming_bandwidth_query,
        "criterion_class": LearningRateBOIncomingBandwidth
    },
    Names.EPOCH_ADJUSTMENT_BASED_ON_CPU_UTILIZATION.value: {
        "query_func": container_specific_cpu_usage_query,
        "criterion_class": EpochAdjustmentBasedOnCPUUtilization
    },
}

STATIC_CONTAINER_CONFIG = {
    Names.CONTAINER_CPU_CORES.value: {
        "query_func": machine_physical_cpu_cores_query
    },
    Names.CONTAINER_MEMORY_LIMIT.value: {
        "query_func": container_memory_limit_query
    }
}

CRITERIA_TO_QUERY_MAPPING = {criteria: (config["query_func"], config["criterion_class"].__name__) for criteria, config in CRITERIA_CONFIG.items()}
CONTAINER_STATIC_QUERIES_MAPPING = {name: config["query_func"] for name, config in STATIC_CONTAINER_CONFIG.items()}
