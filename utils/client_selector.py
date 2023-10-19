import logging
from typing import List, Callable, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
from utils.criteria import AbstractCriterion, MemoryUsageCriterion, MaxCPUUsageCriterion
from config.config_loader import load_criteria_config
from services.prometheus_service import PrometheusService

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

    def filter_clients_by_criteria(self, all_clients: List[ClientProxy], server_round: int, round_timestamps: Dict[int, Dict[str, int]]) -> List[ClientProxy]:
        criteria = self._load_criteria()
        selected_clients = []

        for client in all_clients:
            properties_response = client.get_properties(GetPropertiesIns(config={}), timeout=30)
            client_properties = properties_response.properties

            prev_round_start_time = round_timestamps[server_round - 1].get("start", None)
            prev_round_end_time = round_timestamps[server_round - 1].get("end", None)
            
            # Fetch max CPU usage for the client
            max_cpu_usage = self.prom_service.container_specific_max_cpu_usage(client_properties['container_name'], prev_round_start_time, prev_round_end_time)
            metrics = {"max_cpu_usage": max_cpu_usage}
            logger.info(f"Fetched max CPU usage {max_cpu_usage} for client {client_properties['container_name']}")

            if all(criterion.check(client_properties, metrics) for criterion in criteria):
                selected_clients.append(client)

        logger.info(f"Selected {len(selected_clients)} clients based on criteria")
        return selected_clients

    def _load_criteria(self) -> List[AbstractCriterion]:
        criteria_objects = []
        for crit in self.criteria_config.get('criteria', []):
            if crit['type'] == 'max_cpu_usage':
                criteria_objects.append(MaxCPUUsageCriterion(crit['threshold']))
                logger.info(f"Loaded MaxCPUUsageCriterion with threshold {crit['threshold']}")
            # ... [add other criteria here based on their type] ...
        return criteria_objects
