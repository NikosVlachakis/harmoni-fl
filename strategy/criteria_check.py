import logging
from abc import ABC, abstractmethod
import math
from typing import Dict, Union
from utils.names import Names

logger = logging.getLogger(__name__)

class AbstractCriterion(ABC):
    @abstractmethod
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
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

    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
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

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
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



class LearningRateBasedOnCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        
        self.is_blocking = blocking
        self.active = active
        self.cpu_utilization_threshold = config.get('threshold_cpu_utilization_percentage')  
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')
        
    
    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])
        logger.info(f"cpu usage for client {client_properties.get('container_name')} is {cpu_usase_percentage}%")

        adjusted_learning_rate = client_properties.get('learning_rate')
        current_learning_rate = float(adjusted_learning_rate)

        learning_rate_adjustment = {"learning_rate": current_learning_rate}

        min_learning_rate = 0.0001
        max_learning_rate = 1.0

        # This is to give a weight to the clients that exited (OOM errors)
        extra_weight = 1

        # That means the client exitted (OOM errors) -- give a weight to overcome the exit
        if cpu_usase_percentage == -1 or client_properties.get('container_name') in dropped_out_clients or client_properties.get('container_name') not in round_fit_participant_ids:
            new_learning_rate = current_learning_rate * extra_weight * self.adjustment_factor
            new_learning_rate = min(new_learning_rate, max_learning_rate)
            learning_rate_adjustment["learning_rate"] = new_learning_rate
        
        elif cpu_usase_percentage > self.cpu_utilization_threshold:
            new_learning_rate = current_learning_rate * self.adjustment_factor
            new_learning_rate = min(new_learning_rate, max_learning_rate)
            learning_rate_adjustment["learning_rate"] = new_learning_rate
       
        else:
            new_learning_rate = current_learning_rate / self.adjustment_factor
            new_learning_rate = max(new_learning_rate, min_learning_rate)
            learning_rate_adjustment["learning_rate"] = new_learning_rate
            
        return learning_rate_adjustment


class EpochAdjustmentBasedOnCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])
        logger.info(f"cpu usage for client {client_properties.get('container_name')} is {cpu_usase_percentage}%")

        adjusted_epochs = client_properties.get('epochs')
        current_number_of_epochs = int(adjusted_epochs)

        epoch_adjustment = {"epochs": current_number_of_epochs}
        
        min_epochs = 1
        max_epochs = 5

        # This is to give a weight to the clients that exited (OOM errors)
        extra_weight = 1

        # That means the client exitted (OOM errors) -- give a weight to overcome the exit
        if cpu_usase_percentage == -1 or client_properties.get('container_name') in dropped_out_clients or client_properties.get('container_name') not in round_fit_participant_ids:
            temp_adjusted_epochs = current_number_of_epochs / (extra_weight * self.adjustment_factor)
            if temp_adjusted_epochs < 2:
                adjusted_epochs = 1
            else:
                adjusted_epochs = math.ceil(temp_adjusted_epochs)
            
            epoch_adjustment["epochs"] = max(adjusted_epochs, min_epochs)

        elif cpu_usase_percentage > self.threshold_cpu_utilization_percentage:  
            temp_adjusted_epochs = current_number_of_epochs / self.adjustment_factor
            if temp_adjusted_epochs < 2:
                adjusted_epochs = 1
            else:
                adjusted_epochs = math.ceil(temp_adjusted_epochs)

            epoch_adjustment["epochs"] = max(adjusted_epochs, min_epochs)

        else:
            temp_adjusted_epochs = current_number_of_epochs * self.adjustment_factor
            if temp_adjusted_epochs < 2:
                adjusted_epochs = 1
            else:
                adjusted_epochs = math.ceil(temp_adjusted_epochs)
            
            epoch_adjustment["epochs"] = min(adjusted_epochs, max_epochs)
        
        return epoch_adjustment
    
class AdaptiveBatchSizeBasedOnMemoryUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_memory_utilization_percentage = config.get('threshold_memory_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        average_memory_usage_percentage = float(queries_results['container_memory_usage_percentage'])
        logger.info(f"Average memory usage for client {client_properties.get('container_name')} is {average_memory_usage_percentage}%")

        adjusted_batch_size = client_properties.get('batch_size')
        current_batch_size = int(adjusted_batch_size)
        batch_size_adjustment = {"batch_size": current_batch_size}

        min_batch_size = 10
        max_batch_size = 512
        
        # This is to give a weight to the clients that exited (OOM errors)
        extra_weight = 2.5

        # That means the client exitted (OOM errors) -- give a weight to overcome the exit
        if average_memory_usage_percentage == -1 or client_properties.get('container_name') in dropped_out_clients or client_properties.get('container_name') not in round_fit_participant_ids:
            adjusted_batch_size = math.ceil(current_batch_size / (extra_weight * self.adjustment_factor))
            batch_size_adjustment["batch_size"] = max(adjusted_batch_size, min_batch_size)

        elif average_memory_usage_percentage > self.threshold_memory_utilization_percentage:
            adjusted_batch_size = math.ceil(current_batch_size / self.adjustment_factor)
            batch_size_adjustment["batch_size"] = max(adjusted_batch_size, min_batch_size)

        else:
            adjusted_batch_size = math.ceil(current_batch_size * self.adjustment_factor)
            batch_size_adjustment["batch_size"] = min(adjusted_batch_size, max_batch_size)
        
       

        return batch_size_adjustment
    
class AdaptiveDataSamplingBasedOnMemoryUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_memory_utilization_percentage = config.get('threshold_memory_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
        if not self.active:
            return False

        # Get the average memory usage percentage
        average_memory_usage_percentage = float(queries_results['container_memory_usage_percentage'])
        logger.info(f"Average memory usage for client {client_properties.get('container_name')} is {average_memory_usage_percentage}%")

        # Get the current data sample percentage
        current_data_sample_percentage = float(client_properties.get('data_sample_percentage'))

        data_sample_percentage_adjustment = {"data_sample_percentage": current_data_sample_percentage}

        # Minimum percentage of data to sample
        min_percentage_of_data_to_sample = 0.16
        max_percentage_of_data_to_sample = 1

        # This is to give a weight to the clients that exited (OOM errors)
        extra_weight = 7

        if average_memory_usage_percentage == -1 or client_properties.get('container_name') in dropped_out_clients or client_properties.get('container_name') not in round_fit_participant_ids:
            adjusted_data_sample_percentage = (current_data_sample_percentage / (extra_weight*self.adjustment_factor))
            data_sample_percentage_adjustment["data_sample_percentage"] = max(adjusted_data_sample_percentage, min_percentage_of_data_to_sample)

        elif average_memory_usage_percentage > self.threshold_memory_utilization_percentage:
            adjusted_data_sample_percentage = (current_data_sample_percentage / self.adjustment_factor)
            data_sample_percentage_adjustment["data_sample_percentage"] = max(adjusted_data_sample_percentage, min_percentage_of_data_to_sample)

        else:
            adjusted_data_sample_percentage = (current_data_sample_percentage * self.adjustment_factor)
            data_sample_percentage_adjustment["data_sample_percentage"] = min(adjusted_data_sample_percentage, max_percentage_of_data_to_sample)

        
        return data_sample_percentage_adjustment


class ModelLayerReductionBasedOnHighCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
        if not self.active:
            return False
        
        cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])
        logger.info(f"cpu usage for client {client_properties.get('container_name')} is {cpu_usase_percentage}%")

        current_freeze_percentage = client_properties.get('freeze_layers_percentage')

        current_freeze_percentage = int(current_freeze_percentage) if current_freeze_percentage is not None else 0

        freeze_percentage_adjustment = {"freeze_layers_percentage": current_freeze_percentage}

        min_freeze_percentage = 0
        max_freeze_percentage = 95

        # This is to give a weight to the clients that exited (OOM errors)
        extra_weight = 1

        if cpu_usase_percentage == -1 or client_properties.get('container_name') in dropped_out_clients or client_properties.get('container_name') not in round_fit_participant_ids:
            adjusted_freeze_percentage = math.ceil(current_freeze_percentage + (extra_weight*self.adjustment_factor))
            freeze_percentage_adjustment["freeze_layers_percentage"] = min(adjusted_freeze_percentage, max_freeze_percentage)
        
        elif cpu_usase_percentage > self.threshold_cpu_utilization_percentage:
            adjusted_freeze_percentage = math.ceil(current_freeze_percentage + self.adjustment_factor)
            freeze_percentage_adjustment["freeze_layers_percentage"] = min(adjusted_freeze_percentage, max_freeze_percentage)
        else:
            adjusted_freeze_percentage = math.ceil(current_freeze_percentage - self.adjustment_factor)
            freeze_percentage_adjustment["freeze_layers_percentage"] = max(adjusted_freeze_percentage, min_freeze_percentage)

        return freeze_percentage_adjustment
    
class GradientClippingBasedOnHighCPUUtilization(AbstractCriterion):
    def __init__(self, config: Dict[str, any], blocking: bool, active: bool):
        self.is_blocking = blocking
        self.active = active
        self.threshold_cpu_utilization_percentage = config.get('threshold_cpu_utilization_percentage')
        self.adjustment_factor = config.get('adjustment_factor')
        self.default = config.get('default')

    def check(self, client_properties: Dict[str, str], queries_results: Dict[str, float], dropped_out_clients: list[str], round_fit_participant_ids: list[str]) -> Union[Dict, bool]:
        
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
