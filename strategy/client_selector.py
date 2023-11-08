import logging
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
from strategy.criteria import *
from utils.load_configs import load_criteria_config
from services.prometheus_service import PrometheusService
from config.mappings import *
from typing import Dict, List, Tuple

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
                query = query_func(client_properties['container_name'], prev_round_start_time, prev_round_end_time)
                query_name = query_func.__name__
                if (query, query_name) not in queries:
                    queries.append((query, query_name))
            if not query_funcs:
                logger.error(f"No query function found for criteria type: {crit_type}")

        return queries


    def filter_clients_by_criteria(self, all_clients: List[ClientProxy], server_round: int, round_timestamps: Dict[int, Dict[str, int]]) -> List[ClientProxy]:
        criteria = self._load_criteria()
        selected_clients = []

        for client in all_clients:
            # Get the properties of the client
            properties_response = client.get_properties(GetPropertiesIns(config={}), timeout=30)
            client_properties = properties_response.properties
           
            # Get the start and end time of the previous round
            prev_round_start_time = round_timestamps[server_round - 1].get("start", None)
            prev_round_end_time = round_timestamps[server_round - 1].get("end", None)
            
            # Create a list of queries based on criteria
            queries = self.generate_dynamic_queries_for_client_B0_criteria(client_properties, prev_round_start_time, prev_round_end_time)
            
            # Fetch queries_results for the client
            queries_results = self.prom_service.batch_query(queries)

            # Separate blocking and non-blocking criteria
            blocking_criteria = [c for c in criteria if c.is_blocking]
            non_blocking_criteria = [c for c in criteria if not c.is_blocking]

            # Check each blocking criterion using the fetched queries_results
            if all(criterion.check(client_properties, queries_results) for criterion in blocking_criteria):
                # If all blocking criteria are met, handle non-blocking criteria
                client_config = {}
                for criterion in non_blocking_criteria:
                    result = criterion.check(client_properties, queries_results)
                    if isinstance(result, dict):
                        client_config.update(result)

                selected_client = {
                    'client': client,
                    'config': client_config
                }
                selected_clients.append(selected_client)

        logger.info(f"Selected {len(selected_clients)} clients based on criteria")
        return selected_clients


    def _load_criteria(self) -> List[AbstractCriterion]:
        criteria_objects = []
        # loop through the .yaml file and create criterion objects
        for crit in self.criteria_config.get('criteria', []):
            crit_type = crit.get('type')
            crit_config = crit.get('config', {})
            is_blocking = crit.get('blocking', True)  # Default to True if 'blocking' is not specified
            
            # check if the criterion type is in the mapping
            if crit_type in CRITERIA_CONFIG:
                criterion_config = CRITERIA_CONFIG[crit_type]
                criterion_class = criterion_config.get('criterion_class')

                if criterion_class is not None:
                    criterion_obj = criterion_class(crit_config, is_blocking)
                    criteria_objects.append(criterion_obj)
                else:
                    logger.warning(f"No criterion class found for type {crit_type}")
            else:
                logger.error(f"No configuration found for criteria type: {crit_type}")

        return criteria_objects


