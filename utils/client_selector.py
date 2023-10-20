import logging
from typing import List, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
from services.prometheus_queries import METRIC_TO_QUERY_MAPPING
from utils.criteria import AbstractCriterion, MemoryUsageCriterion, MaxCPUUsageCriterion
from config.config_loader import load_criteria_config
from services.prometheus_service import PrometheusService
from utils.metric_names import MetricNames

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

    # def generate_queries_for_client(self, client_properties: Dict[str, str], prev_round_start_time: int, prev_round_end_time: int) -> List[str]:
    #     queries = []
    #     for crit in self.criteria_config.get('criteria', []):
    #         crit_type = crit.get('type')
    #         logger.info(f"Processing criteria type: {crit_type}")

    #         query_func = METRIC_TO_QUERY_MAPPING.get(crit_type)
    #         if query_func:
    #             query = query_func(client_properties['container_name'], prev_round_start_time, prev_round_end_time)
    #             queries.append(query)
    #             logger.info(f"Generated query for {crit_type}: {query}")
    #         else:
    #             logger.error(f"No query function found for criteria type: {crit_type}")

    #     logger.info(f"Created queries for client {client_properties['container_name']}: {queries}")
    #     return queries
    def generate_queries_for_client(self, client_properties: Dict[str, str], prev_round_start_time: int, prev_round_end_time: int) -> List[tuple]:
        query_tuples = []
        for crit in self.criteria_config.get('criteria', []):
            crit_type = crit.get('type')
            logger.info(f"Processing criteria type: {crit_type}")

            query_func = METRIC_TO_QUERY_MAPPING.get(crit_type)
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

            prev_round_start_time = round_timestamps[server_round - 1].get("start", None)
            prev_round_end_time = round_timestamps[server_round - 1].get("end", None)
            
            # Create a list of queries based on criteria
            query_tuples = self.generate_queries_for_client(client_properties, prev_round_start_time, prev_round_end_time)
            
            # Fetch metrics for the client
            metrics = self.prom_service.batch_query(query_tuples)
            logger.info(f"Fetched metrics for client {client_properties['container_name']}: {metrics}")

            # Check each criterion using the fetched metrics
            if all(criterion.check(client_properties, metrics) for criterion in criteria):
                selected_clients.append(client)

        logger.info(f"Selected {len(selected_clients)} clients based on criteria")
        return selected_clients


    def _load_criteria(self) -> List[AbstractCriterion]:
        criteria_objects = []
        for crit in self.criteria_config.get('criteria', []):
            if crit['type'] == MetricNames.MAX_CPU_USAGE.value:
                criteria_objects.append(MaxCPUUsageCriterion(crit['threshold']))
                logger.info(f"Loaded MaxCPUUsageCriterion with threshold {crit['threshold']}")
            # ... [add other criteria here based on their type] ...
        return criteria_objects
