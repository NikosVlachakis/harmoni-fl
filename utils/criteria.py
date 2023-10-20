import logging

# criteria.py
from abc import ABC, abstractmethod
from typing import Dict
from utils.metric_names import MetricNames

logger = logging.getLogger(__name__)

class AbstractCriterion(ABC):
    @abstractmethod
    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        """Check if the client meets the criteria"""
        pass

class MemoryUsageCriterion(AbstractCriterion):
    def __init__(self, threshold: float):
        self.threshold = threshold
        logger.info(f"Initialized MemoryUsageCriterion with threshold: {threshold}")

    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        meets_criteria = metrics.get("memory_usage", 100) <= self.threshold
        logger.info(f"MemoryUsageCriterion check result: {meets_criteria}")
        return meets_criteria
    
class MaxCPUUsageCriterion(AbstractCriterion):
    def __init__(self, threshold: float):
        self.threshold = threshold
        logger.info(f"Initialized MaxCPUUsageCriterion with threshold: {threshold}")

    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        meets_criteria = float(metrics.get(MetricNames.MAX_CPU_USAGE.value, 100)) <= self.threshold
        logger.info(f"MaxCPUUsageCriterion check result: {meets_criteria}")
        return meets_criteria

class GPUCriterion(AbstractCriterion):
    def check(self, client_properties: Dict[str, str], metrics: Dict[str, float]) -> bool:
        has_gpu = client_properties.get("has_gpu", False)
        logger.info(f"GPUCriterion check result: {has_gpu}")
        return has_gpu
