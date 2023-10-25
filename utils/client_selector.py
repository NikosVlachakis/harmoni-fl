import logging
from typing import List, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
from utils.criteria import *
from config.config_loader import load_criteria_config
from services.prometheus_service import PrometheusService
from config.mappings import *

logger = logging.getLogger(__name__)

class ClientSelector:
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager
        self.criteria_config = load_criteria_config("config/criteria.yaml")
        self.prom_service = PrometheusService()
        logger.info("ClientSelector initialized")

    def get_all_clients(self) -> List[ClientProxy]:
        total_available_clients = self.client_manager.num_available()
        clients = self.client_manager.sample(num_clients=total_available_clients, min_num_clients=total_available_clients)
        logger.info(f"Retrieved {len(clients)} clients")
        return clients

    
    def generate_dynamic_queries_for_client_B0_criteria(self, client_properties: Dict[str, str], prev_round_start_time: int, prev_round_end_time: int) -> List[tuple]:
        query_tuples = []
        for crit in self.criteria_config.get('criteria', []):
            crit_type = crit.get('type')
            logger.info(f"Processing criteria type: {crit_type}")

            query_func = CRITERIA_TO_QUERY_MAPPING.get(crit_type)
            if query_func:
                query = query_func(client_properties['container_name'], prev_round_start_time, prev_round_end_time)
                query_tuples.append((query, query_func.__name__))
                logger.info(f"Generated query for {crit_type}: {query}")
            else:
                logger.error(f"No query function found for criteria type: {crit_type}")

        logger.info(f"Created queries for client {client_properties['container_name']}: {query_tuples}")
        return query_tuples


    def filter_clients_by_criteria(self, all_clients: List[ClientProxy], server_round: int, round_timestamps: Dict[int, Dict[str, int]]) -> List[ClientProxy]:
        criteria = self._load_criteria()
        selected_clients = []

        for client in all_clients:
            properties_response = client.get_properties(GetPropertiesIns(config={}), timeout=30)
            client_properties = properties_response.properties

            # Get the start and end time of the previous round
            prev_round_start_time = round_timestamps[server_round - 1].get("start", None)
            prev_round_end_time = round_timestamps[server_round - 1].get("end", None)
            
            # Create a list of queries based on criteria
            query_tuples = self.generate_dynamic_queries_for_client_B0_criteria(client_properties, prev_round_start_time, prev_round_end_time)
            
            # Fetch metrics for the client
            metrics = self.prom_service.batch_query(query_tuples)

            # Separate blocking and non-blocking criteria
            blocking_criteria = [c for c in criteria if c.is_blocking]
            non_blocking_criteria = [c for c in criteria if not c.is_blocking]

            # Check each blocking criterion using the fetched metrics
            if all(criterion.check(client_properties, metrics) for criterion in blocking_criteria):
                # If all blocking criteria are met, handle non-blocking criteria
                client_config = {}
                for criterion in non_blocking_criteria:
                    result = criterion.check(client_properties, metrics)
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
        for crit in self.criteria_config.get('criteria', []):
            crit_type = crit.get('type')
            crit_config = crit.get('config', {})
            is_blocking = crit.get('blocking', True)  # Default to True if 'blocking' is not specified

            if crit_type in CRITERIA_CONFIG:
                criterion_config = CRITERIA_CONFIG[crit_type]
                criterion_class = criterion_config.get('criterion_class')

                if criterion_class is not None:
                    criterion_obj = criterion_class(crit_config, is_blocking)
                    criteria_objects.append(criterion_obj)
                    logger.info(f"Loaded {criterion_class.__name__} with config {crit_config} and blocking={is_blocking}")
                else:
                    logger.warning(f"No criterion class found for type {crit_type}")
            else:
                logger.error(f"No configuration found for criteria type: {crit_type}")

        return criteria_objects


