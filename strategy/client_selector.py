import logging
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from strategy.criteria_check import *
from utils.simple_utils import load_criteria_config
from services.prometheus_service import PrometheusService
from helpers.mappings import *
from typing import Dict, List, Tuple
from utils.simple_utils import get_client_properties

logger = logging.getLogger(__name__)

class ClientSelector:
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager
        self.criteria_config = load_criteria_config("config/criteria.yaml")
        self.prom_service = PrometheusService()

    def get_all_clients(self) -> List[ClientProxy]:
        total_available_clients = self.client_manager.num_available()
        clients = self.client_manager.sample(num_clients=total_available_clients, min_num_clients=total_available_clients)
        return clients

    
    def generate_dynamic_queries_for_client_B0_criteria(self, client_properties: Dict[str, str], prev_round_start_time: int, prev_round_end_time: int) -> List[Tuple[str, str]]:
        queries = []
        for crit in self.criteria_config.get('criteria', []):
            crit_type = crit.get('type')

            query_funcs = CRITERIA_TO_QUERY_MAPPING.get(crit_type, [])

            for query_func in query_funcs:
                query = query_func(container_name = client_properties['container_name'], start_timestamp = prev_round_start_time , end_timestamp = prev_round_end_time)
                query_name = query_func.__name__
                if (query, query_name) not in queries:
                    queries.append((query, query_name))
            if not query_funcs:
                logger.error(f"No query function found for criteria type: {crit_type}")

        return queries


    def filter_clients_by_criteria(self, all_clients: List[ClientProxy], round_timestamps: Dict[str, Dict[str, float]], dropped_out_clients: list[str]) -> List[ClientProxy]:
        # Get the criteria objects
        criteria_objects = self._load_criteria()
        
        # If no criteria are specified, return all clients
        if not criteria_objects:
            logger.info("No criteria specified, returning an empty list of selected clients")
            return []

        # Initialize the list of selected clients
        selected_clients = []

        for client in all_clients:
            # Get the properties of the client
            client_properties = get_client_properties(client)
                        
            # Get the start and end timesmtamps for the current client
            # prev_round_start_time = round_timestamps[client_properties['container_name']].get("start", None)
            # prev_round_end_time = round_timestamps[client_properties['container_name']].get("end", None)
            
            try:
                # Attempt to get the start and end timestamps for the current client
                prev_round_start_time = round_timestamps[client_properties['container_name']].get("start", None)
                prev_round_end_time = round_timestamps[client_properties['container_name']].get("end", None)

                # Proceed only if both timestamps are available, indicating participation in a previous round
                if prev_round_start_time is None or prev_round_end_time is None:
                    raise KeyError(f"Client {client_properties['container_name']} does not have valid timestamps for a previous round.")
            
            except KeyError as e:
                # For first-time participants, assume default or no specific configuration needed
                logger.info(f"Client {client_properties['container_name']} is considered a new participant.")
                selected_clients.append({
                    'client': client,
                    'config': {
                        "epochs": 1,
                        "batch_size": 16,
                        "learning_rate": 0.01,
                        "data_sample_percentage": 0.08,
                        "freeze_layers_percentage": 0,
                    }
                })
                continue  # Proceed to the next client without further processing


            # Create a list of queries based on criteria
            queries = self.generate_dynamic_queries_for_client_B0_criteria(client_properties, prev_round_start_time, prev_round_end_time)
            
            # Fetch queries_results for the client
            queries_results = self.prom_service.batch_query(queries)

            # Separate blocking and non-blocking criteria
            blocking_criteria = [c for c in criteria_objects if c.is_blocking]
            non_blocking_criteria = [c for c in criteria_objects if not c.is_blocking]

            # Check each blocking criterion using the fetched queries_results
            if all(criterion.check(client_properties, queries_results, dropped_out_clients) for criterion in blocking_criteria):
                # If all blocking criteria are met, handle non-blocking criteria
                client_config = {}
                for criterion in non_blocking_criteria:
                    result = criterion.check(client_properties, queries_results, dropped_out_clients)
                    if result and isinstance(result, dict):
                        client_config.update(result)

                ############# Add just for experiments the cpu usage and memory usage to the client config #############
                average_memory_usage_percentage = float(queries_results['container_memory_usage_percentage'])
                cpu_usase_percentage = float(queries_results['container_cpu_usage_percentage'])
                    
                client_config.update({
                    'cpu_usase_percentage': cpu_usase_percentage,
                    'average_memory_usage_percentage': average_memory_usage_percentage
                })
                ############# Add just for experiments the cpu usage and memory usage to the client config #############
                
                selected_client = {
                    'client': client,
                    'config': client_config
                }
                selected_clients.append(selected_client)

        logger.info(f"Selected {len(selected_clients)} clients based on criteria")
        return selected_clients


    def _load_criteria(self) -> List[AbstractCriterion]:
        criteria_objects = []
        criteria_config = self.criteria_config.get('criteria', [])

         # Handle the case where criteria_config is None
        if criteria_config is None:
            logger.warning("No criteria found in the configuration")
            return criteria_objects


        # loop through the .yaml file and create criterion objects
        for crit in criteria_config:
            crit_type = crit.get('type')
            crit_config = crit.get('config', {})
            is_blocking = crit.get('blocking', True)
            is_active = crit.get('active', True)

            # check if the criterion type is in the mapping
            if crit_type in CRITERIA_CONFIG:
                criterion_config = CRITERIA_CONFIG[crit_type]
                criterion_class = criterion_config.get('criterion_class')

                if criterion_class is not None:
                    criterion_obj = criterion_class(crit_config, is_blocking, is_active)
                    criteria_objects.append(criterion_obj)
                else:
                    logger.warning(f"No criterion class found for type {crit_type}")
            else:
                logger.error(f"No configuration found for criteria type: {crit_type}")

        return criteria_objects


