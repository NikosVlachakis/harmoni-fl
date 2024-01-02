from strategy.criteria import *
from utils.names import Names
from services.prometheus_queries import *

CRITERIA_CONFIG = {
    Names.LEARNING_RATE_BASED_ON_INCOMING_BANDWIDTH.value: {
        "query_func": [container_incoming_bandwidth_query],
        "criterion_class": LearningRateBOIncomingBandwidth
    },
    Names.EPOCH_ADJUSTMENT_BASED_ON_CPU_UTILIZATION.value: {
        "query_func": [container_cpu_usage_percentage],
        "criterion_class": EpochAdjustmentBasedOnCPUUtilization
    },
    Names.ADAPTIVE_BATCH_SIZE_BASED_ON_MEMORY_UTILIZATION.value: {
        "query_func": [container_memory_usage_percentage],
        "criterion_class": AdaptiveBatchSizeBasedOnMemoryUtilization
    },
    Names.ADAPTIVE_DATA_SAMPLING_BASED_ON_MEMORY_UTILIZATION.value: {
        "query_func": [container_memory_usage_percentage],
        "criterion_class": AdaptiveDataSamplingBasedOnMemoryUtilization
    },
    Names.MODEL_LAYER_FREEZING_BASED_ON_HIGH_CPU_UTILIZATION.value: {
        "query_func": [container_cpu_usage_percentage],
        "criterion_class": ModelLayerReductionBasedOnHighCPUUtilization
    },
    Names.GRADIENT_CLIPPING_BASED_ON_HIGH_CPU_UTILIZATION.value: {
        "query_func": [container_cpu_usage_percentage],
        "criterion_class": GradientClippingBasedOnHighCPUUtilization
    },
    Names.WEIGHT_PRECISION_BASED_ON_HIGH_CPU_UTILIZATION.value: {
        "query_func": [container_cpu_usage_percentage],
        "criterion_class": ModelPrecisionBasedOnHighCPUUtilization
    },
    Names.INCLUDE_CLIENTS_WITHIN_SPECIFIC_THRESHOLDS.value: {
        "query_func": [container_cpu_usage_percentage,container_memory_usage_percentage],
        "criterion_class": IncludeClientsWithinSpecificThresholds
    },
    Names.SPARSIFICATION_BASED_ON_OUTGOING_BANDWIDTH.value: {
        "query_func": [container_outgoing_bandwidth_query],
        "criterion_class": SparsificationBOOutgoingBandwidth
    }
}


CRITERIA_TO_QUERY_MAPPING = {criteria: config["query_func"] for criteria, config in CRITERIA_CONFIG.items()}
