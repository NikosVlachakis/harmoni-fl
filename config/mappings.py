from utils.criteria import *
from utils.names import Names
from services.prometheus_queries import *

CRITERIA_CONFIG = {
    Names.MAX_CPU_USAGE.value: {
        "query_func": container_specific_max_cpu_usage_query,
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
}

CRITERIA_TO_QUERY_MAPPING = {criteria: config["query_func"] for criteria, config in CRITERIA_CONFIG.items()}
