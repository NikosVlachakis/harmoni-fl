import requests
import json
from services.prometheus_queries import *
import logging
from typing import Dict, List
from config.mappings import *

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class PrometheusService:
    def __init__(self):
        self.prom_url = 'http://host.docker.internal:9090'

    def query(self, query):
        response = requests.get(f'{self.prom_url}/api/v1/query', params={'query': query})
        data = response.json()
        return data['data']['result'][0]['value'][1]
    
    
    def batch_query(self, queries: List[tuple]) -> Dict[str, float]:
        results = {}
        # criterion_class acts as the identifier for the criteria
        for query, query_name in queries:
            try:
                result = self.query(query)
                results[query_name] = float(result)  # Ensure the result is converted to a float
            except Exception as e:
                logger.error(f"Error while fetching value for query {query}: {e}")
        return results
    
    
    def get_container_static_metrics(self, container_name, queries_mapping=CONTAINER_STATIC_QUERIES_MAPPING):
        results = {}
        for metric_name, query_func in queries_mapping.items():
            try:
                query = query_func(container_name)
                result = self.query(query)
                results[metric_name] = float(result)
            except Exception as e:
                logger.error(f"Error fetching result for query {query}: {e}")
        return results






