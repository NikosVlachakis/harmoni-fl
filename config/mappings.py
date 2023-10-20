from utils.criteria import MaxCPUUsageCriterion, MAXMemoryUsageCriterion
from utils.metric_names import MetricNames
from services.prometheus_queries import *

CRITERIA_CONFIG = {
    MetricNames.MAX_CPU_USAGE.value: {
        "query_func": container_specific_max_cpu_usage_query,
        "criterion_class": MaxCPUUsageCriterion
    },
    MetricNames.MAX_MEMORY_USAGE_PERCENTAGE.value: {
        "query_func": container_specific_max_memory_usage_query,
        "criterion_class": MAXMemoryUsageCriterion
    }
}

METRIC_TO_QUERY_MAPPING = {metric: config["query_func"] for metric, config in CRITERIA_CONFIG.items()}
