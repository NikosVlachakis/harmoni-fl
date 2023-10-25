import logging
from abc import ABC, abstractmethod
import math
from typing import Dict
from utils.names import Names

logger = logging.getLogger(__name__)

# The idea behind check method is that some criteria may be blocking for the selection process, 
# i.e. if the client does not meet the criteria, it should not be selected at all.
# There is going to be the case that some criteria are not blocking, and just provide the custom configuration
# for each client for the round.

class AbstractCriterion(ABC):
    @abstractmethod
    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        """Check if the client meets the criteria"""
        pass

class MAXMemoryUsageCriterion(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.threshold = config.get('threshold')  # Default threshold to 100 if not provided
        self.is_blocking = blocking
        logger.info(f"Initialized MAXMemoryUsageCriterion with threshold: {self.threshold}")

    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        percentage_memory_consumed = (float(metrics.get(Names.MAX_MEMORY_USAGE_PERCENTAGE.value, 0)) / float(client_properties.get(Names.CONTAINER_MEMORY_LIMIT.value))) * 100
        logger.info(f"Percentage memory consumed: {percentage_memory_consumed}")
        meets_criteria = percentage_memory_consumed <= self.threshold
        logger.info(f"MAXMemoryUsageCriterion check result: {meets_criteria}")
        return meets_criteria
    
class MaxCPUUsageCriterion(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.threshold = config.get('threshold')  # Default threshold to 100 if not provided
        self.is_blocking = blocking
        logger.info(f"Initialized MaxCPUUsageCriterion with threshold: {self.threshold}")

    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        meets_criteria = float(metrics.get(Names.MAX_CPU_USAGE.value, 0)) <= self.threshold
        logger.info(f"MaxCPUUsageCriterion check result: {meets_criteria}")
        return meets_criteria

class GPUCriterion(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        pass  # No configuration required for GPU criterion at the moment

    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        has_gpu = client_properties.get("has_gpu", False)
        logger.info(f"GPUCriterion check result: {has_gpu}")
        return has_gpu
    
class LearningRateBOIncomingBandwidth(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.bandwidth_threshold = config.get('threshold_bandwidth_mbps', 10)  # in Mbps, default to 10Mbps if not provided
        self.adjustment_factor = config.get('adjustment_factor', 1.5)  # default to 1.5 if not provided
        self.default_learning_rate = config.get('default_learning_rate', 0.01)  # default to 0.01 if not provided
        self.is_blocking = blocking
        logger.info(f"Initialized LearningRateBOIncomingBandwidth with bandwidth_threshold: {self.bandwidth_threshold} Mbps, adjustment_factor: {self.adjustment_factor}, default_learning_rate: {self.default_learning_rate}")
    
    def check(self, client_properties: Dict[str, str], metrics: Dict[str, any]) -> Dict[str, any]:
        incoming_bandwidth = float(metrics.get(Names.LEARNING_RATE_BASED_ON_INCOMING_BANDWIDTH.value))  # in Mbps
        logger.info(f"LearningRateBOIncomingBandwidth check result: {incoming_bandwidth} Mbps")
        learning_rate_adjustment = {"learning_rate": self.default_learning_rate}

        if incoming_bandwidth < self.bandwidth_threshold:
            adjusted_learning_rate = self.default_learning_rate * self.adjustment_factor
            learning_rate_adjustment["learning_rate"] = adjusted_learning_rate
            logger.info(f"Adjusted learning rate to {adjusted_learning_rate} due to low incoming bandwidth ({incoming_bandwidth} Mbps)")

        return learning_rate_adjustment

class EpochAdjustmentBasedOnCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.default_number_of_epochs = config.get('default_number_of_epochs')
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
    
    def check(self, client_properties: Dict[str, str], metrics: Dict[str, any]) -> Dict[str, any]:
        container_cpu_cores = float(client_properties.get(Names.CONTAINER_CPU_CORES.value, 4))
        rate_of_cpu_usase = float(metrics.get(Names.EPOCH_ADJUSTMENT_BASED_ON_CPU_UTILIZATION.value))
        cpu_utlization = (rate_of_cpu_usase / container_cpu_cores) * 100
        logger.info(f"EpochAdjustmentBasedOnCPUUtilization check result: {cpu_utlization}%")
        epoch_adjustment = {"epochs": self.default_number_of_epochs}

        if cpu_utlization > self.threshold_cpu_utilization_percentage:
            epoch_adjustment["epochs"] = math.floor(self.default_number_of_epochs / self.adjustment_factor)
            logger.info(f"Adjusted number of epochs to {epoch_adjustment['epochs']} due to high CPU utilization of ({cpu_utlization}%)")

        return epoch_adjustment