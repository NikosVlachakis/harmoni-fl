import logging
from abc import ABC, abstractmethod
import math
from typing import Dict, Union
from utils.names import Names

logger = logging.getLogger(__name__)

class AbstractCriterion(ABC):
    @abstractmethod
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        """Check if the client meets the criteria"""
        pass

class IncludeClientsWithinSpecificThresholds(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.min_cpu_utilization_percentage = config.get('min_cpu_utilization_percentage')
        self.max_cpu_utilization_percentage = config.get('max_cpu_utilization_percentage')
        self.min_memory_utilization_percentage = config.get('min_memory_utilization_percentage')
        self.max_memory_utilization_percentage = config.get('max_memory_utilization_percentage')

    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return True
        
        cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])

        average_memory_usage_percentage = float(queries_results['container_memory_usage_percentage'])

        meets_criteria = (self.min_cpu_utilization_percentage <= cpu_usase_percentage <= self.max_cpu_utilization_percentage) and (self.min_memory_utilization_percentage <= average_memory_usage_percentage <= self.max_memory_utilization_percentage)

        if not meets_criteria:
            logger.info(f"Client {client_properties.get('container_name')} does not meet the criteria because CPU utilization is {cpu_usase_percentage}% and memory utilization is {average_memory_usage_percentage}%")
        else:
            logger.info(f"Client {client_properties.get('container_name')} meets the criteria because CPU utilization is {cpu_usase_percentage}% and memory utilization is {average_memory_usage_percentage}%")

        return meets_criteria

class SparsificationBOOutgoingBandwidth(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_bandwidth_MBps = config.get('threshold_bandwidth_MBps')
        self.default = config.get('default')
        self.methods = config.get('methods', {})

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return False
        
        container_outgoing_bandwidth_query = float(queries_results.get('container_outgoing_bandwidth_query')) # in Mbps

        if container_outgoing_bandwidth_query < self.threshold_bandwidth_MBps:
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
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        
        self.is_blocking = blocking
        self.active = active
        self.bandwidth_threshold = config.get('threshold_bandwidth_MBps')  
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')
        
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        container_incoming_bandwidth_query = float(queries_results['container_incoming_bandwidth_query'])  # in Mbps
        
        adjusted_learning_rate = client_properties.get('learning_rate')
        current_learning_rate = float(adjusted_learning_rate)

        learning_rate_adjustment = {"learning_rate": current_learning_rate}

        if container_incoming_bandwidth_query < self.bandwidth_threshold:
            new_learning_rate = current_learning_rate * self.adjustment_factor
            learning_rate_adjustment["learning_rate"] = new_learning_rate
            logger.info(f"Adjusted learning rate to {new_learning_rate} due to low incoming bandwidth ({container_incoming_bandwidth_query} Mbps)")
       
        else:
            learning_rate_adjustment["learning_rate"] = self.default
            
        return learning_rate_adjustment


class EpochAdjustmentBasedOnCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])
        logger.info(f"cpu usage for client {client_properties.get('container_name')} is {cpu_usase_percentage}%")

        adjusted_epochs = client_properties.get('epochs')
        current_number_of_epochs = int(adjusted_epochs)

        epoch_adjustment = {"epochs": current_number_of_epochs}

        if cpu_usase_percentage > self.threshold_cpu_utilization_percentage:
            adjusted_epochs = math.ceil(current_number_of_epochs / self.adjustment_factor)
            epoch_adjustment["epochs"] = max(adjusted_epochs, 1)
            logger.info(f"Adjusted number of epochs to {epoch_adjustment['epochs']} due to high cpu usase of ({cpu_usase_percentage}%)")

        else:
            epoch_adjustment["epochs"] = self.default
        
        return epoch_adjustment
    
class AdaptiveBatchSizeBasedOnMemoryUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_memory_utilization_percentage = config.get('threshold_memory_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        average_memory_usage_percentage = float(queries_results['container_memory_usage_percentage'])
        logger.info(f"Average memory usage for client {client_properties.get('container_name')} is {average_memory_usage_percentage}%")

        adjusted_batch_size = client_properties.get('batch_size')
        current_batch_size = int(adjusted_batch_size)
        batch_size_adjustment = {"batch_size": current_batch_size}

        if average_memory_usage_percentage > self.threshold_memory_utilization_percentage:
            adjusted_batch_size = math.ceil(current_batch_size / self.adjustment_factor)
            batch_size_adjustment["batch_size"] = max(adjusted_batch_size, 1)
            logger.info(f"Adjusted batch size to {batch_size_adjustment['batch_size']} due to high memory utilization of ({average_memory_usage_percentage}%)")

        else:
            batch_size_adjustment["batch_size"] = self.default
        
        return batch_size_adjustment
    
class AdaptiveDataSamplingBasedOnMemoryUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_memory_utilization_percentage = config.get('threshold_memory_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        # Get the average memory usage percentage
        average_memory_usage_percentage = float(queries_results['container_memory_usage_percentage'])
        logger.info(f"Average memory usage for client {client_properties.get('container_name')} is {average_memory_usage_percentage}%")

        # Get the current data sample percentage
        current_data_sample_percentage = float(client_properties.get('data_sample_percentage'))

        data_sample_percentage_adjustment = {"data_sample_percentage": current_data_sample_percentage}

        if average_memory_usage_percentage > self.threshold_memory_utilization_percentage:
            
            adjusted_data_sample_percentage = (current_data_sample_percentage / self.adjustment_factor)
            
            # Ensure that the data sample percentage is not less than 1%
            data_sample_percentage_adjustment["data_sample_percentage"] = max(adjusted_data_sample_percentage, 0.01)
            logger.info(f"Adjusted data sample percentage to {data_sample_percentage_adjustment['data_sample_percentage']} due to high memory utilization of ({average_memory_usage_percentage}%)")

        else:
            data_sample_percentage_adjustment["data_sample_percentage"] = self.default
        
        return data_sample_percentage_adjustment


class ModelLayerReductionBasedOnHighCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return False
        
        cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])
        logger.info(f"cpu usage for client {client_properties.get('container_name')} is {cpu_usase_percentage}%")

        current_freeze_percentage = client_properties.get('freeze_layers_percentage')

        current_freeze_percentage = int(current_freeze_percentage) if current_freeze_percentage is not None else 0

        freeze_percentage_adjustment = {"freeze_layers_percentage": current_freeze_percentage}


        if cpu_usase_percentage > self.threshold_cpu_utilization_percentage:
            adjusted_freeze_percentage = math.ceil(current_freeze_percentage + self.adjustment_factor)
            freeze_percentage_adjustment["freeze_layers_percentage"] = min(adjusted_freeze_percentage, 90)
            logger.info(f"Adjusted percentage of freezing layers to {freeze_percentage_adjustment['freeze_layers_percentage']} due to high CPU utilization of ({cpu_usase_percentage}%)")
        else:
            freeze_percentage_adjustment["freeze_layers_percentage"] = self.default

        return freeze_percentage_adjustment
    
class GradientClippingBasedOnHighCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])

        current_gradient_clipping = client_properties.get('gradient_clipping_value')

        current_gradient_clipping = float(current_gradient_clipping) if current_gradient_clipping is not None else 0

        gradient_clipping_adjustment = {"gradient_clipping_value": current_gradient_clipping}

        if cpu_usase_percentage > self.threshold_cpu_utilization_percentage:
            adjusted_gradient_clipping = float(current_gradient_clipping - self.adjustment_factor)
            gradient_clipping_adjustment["gradient_clipping_value"] = max(adjusted_gradient_clipping, 0.5)
            logger.info(f"Adjusted gradient clipping to {gradient_clipping_adjustment['gradient_clipping_value']} due to high CPU utilization of ({cpu_usase_percentage}%)")

        else:
            gradient_clipping_adjustment["gradient_clipping_value"] = self.default
        
        return gradient_clipping_adjustment
