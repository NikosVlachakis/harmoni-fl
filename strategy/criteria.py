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

class IncludeClientsWithinSpecificThresholds(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.min_cpu_utilization_percentage = config.get('min_cpu_utilization_percentage')
        self.max_cpu_utilization_percentage = config.get('max_cpu_utilization_percentage')
        self.min_memory_utilization_percentage = config.get('min_memory_utilization_percentage')
        self.max_memory_utilization_percentage = config.get('max_memory_utilization_percentage')

    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        
        container_cpu_cores = float(client_properties.get(Names.CONTAINER_CPU_CORES.value, 4))
        rate_of_cpu_usase = float(queries_results['container_specific_rate_of_cpu_usase_query'])
        
        cpu_utlization = (rate_of_cpu_usase / container_cpu_cores) * 100
        
        average_memory_usage = float(queries_results['container_specific_average_memory_usage_query'])
        percentage_memory_consumed = (average_memory_usage / float(client_properties.get(Names.CONTAINER_MEMORY_LIMIT.value))) * 100
        
        meets_criteria = (self.min_cpu_utilization_percentage <= cpu_utlization <= self.max_cpu_utilization_percentage) and (self.min_memory_utilization_percentage <= percentage_memory_consumed <= self.max_memory_utilization_percentage)

        if not meets_criteria:
            logger.info(f"Client {client_properties.get('container_name')} does not meet the criteria because CPU utilization is {cpu_utlization}% and memory utilization is {percentage_memory_consumed}%")
        else:
            logger.info(f"Client {client_properties.get('container_name')} meets the criteria because CPU utilization is {cpu_utlization}% and memory utilization is {percentage_memory_consumed}%")

        return meets_criteria

class SparsificationBOOutgoingBandwidth(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.threshold_bandwidth_mbps = config.get('threshold_bandwidth_mbps')
        self.methods = config.get('methods', {})

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        container_outgoing_bandwidth_query = float(queries_results.get('container_outgoing_bandwidth_query')) # in Mbps

        if container_outgoing_bandwidth_query < self.threshold_bandwidth_mbps:
            for method_name, method_details in self.methods.items():
                if method_details['enabled']:
                    sparsification_config = {
                        "sparsification_enabled": True,
                        "sparsification_method": method_name,
                        "sparsification_percentile": method_details.get('percentile'),
                    }
                logger.info(f"Adjusted sparsification config for client {client_properties.get('container_name')} as follows: {sparsification_config} because outgoing bandwidth is {container_outgoing_bandwidth_query} Mbps")
                return sparsification_config

        # Default configuration if bandwidth is not below threshold or no method is enabled
        return {
            "sparsification_enabled": False,
            "sparsification_method": None,
            "sparsification_percentile": None,
        }



class LearningRateBOIncomingBandwidth(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.bandwidth_threshold = config.get('threshold_bandwidth_mbps')  
        self.adjustment_factor = config.get('adjustment_factor')
        self.is_blocking = blocking
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        
        container_incoming_bandwidth_query = float(queries_results['container_incoming_bandwidth_query'])  # in Mbps
        
        adjusted_learning_rate = client_properties.get('learning_rate')
        current_learning_rate = float(adjusted_learning_rate)

        learning_rate_adjustment = {"learning_rate": current_learning_rate}

        if container_incoming_bandwidth_query < self.bandwidth_threshold:
            new_learning_rate = current_learning_rate * self.adjustment_factor
            learning_rate_adjustment["learning_rate"] = new_learning_rate
            logger.info(f"Adjusted learning rate to {new_learning_rate} due to low incoming bandwidth ({container_incoming_bandwidth_query} Mbps)")

        return learning_rate_adjustment


class EpochAdjustmentBasedOnCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        
        container_cpu_cores = float(client_properties.get(Names.CONTAINER_CPU_CORES.value, 4))
        rate_of_cpu_usase = float(queries_results['container_specific_rate_of_cpu_usase_query'])
        cpu_utlization = (rate_of_cpu_usase / container_cpu_cores) * 100
        
        adjusted_epochs = client_properties.get('epochs')
        current_number_of_epochs = int(adjusted_epochs)

        epoch_adjustment = {"epochs": current_number_of_epochs}

        if cpu_utlization > self.threshold_cpu_utilization_percentage:
            adjusted_epochs = math.ceil(current_number_of_epochs / self.adjustment_factor)
            epoch_adjustment["epochs"] = max(adjusted_epochs, 1)
            logger.info(f"Adjusted number of epochs to {epoch_adjustment['epochs']} due to high CPU utilization of ({cpu_utlization}%)")

        return epoch_adjustment
    
class AdaptiveBatchSizeBasedOnMemoryUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.threshold_memory_utilization_percentage = config.get('threshold_memory_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        
        average_memory_usage = float(queries_results['container_specific_average_memory_usage_query'])
        percentage_memory_consumed = (average_memory_usage / float(client_properties.get(Names.CONTAINER_MEMORY_LIMIT.value))) * 100
        
        adjusted_batch_size = client_properties.get('batch_size')
        current_batch_size = int(adjusted_batch_size)
        batch_size_adjustment = {"batch_size": current_batch_size}

        if percentage_memory_consumed > self.threshold_memory_utilization_percentage:
            adjusted_batch_size = math.ceil(current_batch_size / self.adjustment_factor)
            batch_size_adjustment["batch_size"] = max(adjusted_batch_size, 1)
            logger.info(f"Adjusted batch size to {batch_size_adjustment['batch_size']} due to high memory utilization of ({percentage_memory_consumed}%)")

        return batch_size_adjustment
    
class AdaptiveDataSamplingBasedOnMemoryUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.threshold_memory_utilization_percentage = config.get('threshold_memory_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        
        average_memory_usage = float(queries_results['container_specific_average_memory_usage_query'])
        percentage_memory_consumed = (average_memory_usage / float(client_properties.get(Names.CONTAINER_MEMORY_LIMIT.value))) * 100
        
        adjusted_data_sample_percentage = client_properties.get('data_sample_percentage')

        current_data_sample_percentage = float(adjusted_data_sample_percentage)

        data_sample_percentage_adjustment = {"data_sample_percentage": current_data_sample_percentage}

        if percentage_memory_consumed > self.threshold_memory_utilization_percentage:
            adjusted_data_sample_percentage = (current_data_sample_percentage / self.adjustment_factor)
            # Ensure that the data sample percentage is not less than 5%
            data_sample_percentage_adjustment["data_sample_percentage"] = max(adjusted_data_sample_percentage, 0.05)
            logger.info(f"Adjusted data sample percentage to {data_sample_percentage_adjustment['data_sample_percentage']} due to high memory utilization of ({percentage_memory_consumed}%)")

        return data_sample_percentage_adjustment


class ModelLayerReductionBasedOnHighCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        
        container_cpu_cores = float(client_properties.get(Names.CONTAINER_CPU_CORES.value, 4))
        rate_of_cpu_usase = float(queries_results['container_specific_rate_of_cpu_usase_query'])
        
        cpu_utlization = (rate_of_cpu_usase / container_cpu_cores) * 100
        
        current_freeze_percentage = client_properties.get('freeze_layers_percentage')

        current_freeze_percentage = int(current_freeze_percentage) if current_freeze_percentage is not None else 0

        freeze_percentage_adjustment = {"freeze_layers_percentage": current_freeze_percentage}


        if cpu_utlization > self.threshold_cpu_utilization_percentage:
            adjusted_freeze_percentage = math.ceil(current_freeze_percentage + self.adjustment_factor)
            freeze_percentage_adjustment["freeze_layers_percentage"] = min(adjusted_freeze_percentage, 90)
            logger.info(f"Adjusted percentage of freezing layers to {freeze_percentage_adjustment['freeze_layers_percentage']} due to high CPU utilization of ({cpu_utlization}%)")

        return freeze_percentage_adjustment
    
class GradientClippingBasedOnHighCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        container_cpu_cores = float(client_properties.get(Names.CONTAINER_CPU_CORES.value, 4))
        rate_of_cpu_usase = float(queries_results['container_specific_rate_of_cpu_usase_query'])
        
        cpu_utlization = (rate_of_cpu_usase / container_cpu_cores) * 100
        
        current_gradient_clipping = client_properties.get('gradient_clipping_value')

        current_gradient_clipping = float(current_gradient_clipping) if current_gradient_clipping is not None else 0

        gradient_clipping_adjustment = {"gradient_clipping_value": current_gradient_clipping}

        if cpu_utlization > self.threshold_cpu_utilization_percentage:
            adjusted_gradient_clipping = float(current_gradient_clipping - self.adjustment_factor)
            gradient_clipping_adjustment["gradient_clipping_value"] = max(adjusted_gradient_clipping, 0.5)
            logger.info(f"Adjusted gradient clipping to {gradient_clipping_adjustment['gradient_clipping_value']} due to high CPU utilization of ({cpu_utlization}%)")

        return gradient_clipping_adjustment

class ModelPrecisionBasedOnHighCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool):
        self.is_blocking = blocking
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.new_precision_value = config.get('new_precision_value')
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Dict[str, any]:
        
        container_cpu_cores = float(client_properties.get(Names.CONTAINER_CPU_CORES.value, 4))
        rate_of_cpu_usase = float(queries_results['container_specific_rate_of_cpu_usase_query'])
        
        cpu_utlization = (rate_of_cpu_usase / container_cpu_cores) * 100
        
        current_model_precision = client_properties.get('model_precision')

        current_model_precision = str(current_model_precision) if current_model_precision is not None else "float32"

        model_precision_adjustment = {"model_precision": current_model_precision}

        if cpu_utlization > self.threshold_cpu_utilization_percentage:
            model_precision_adjustment["model_precision"] = self.new_precision_value
            logger.info(f"Adjusted model precision to {model_precision_adjustment['model_precision']} due to high CPU utilization of ({cpu_utlization}%)")

        return model_precision_adjustment