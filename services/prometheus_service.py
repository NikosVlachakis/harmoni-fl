import requests
import json
from services.prometheus_queries import *
import logging
from typing import Dict, List
from config.mappings import METRIC_TO_QUERY_MAPPING

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class PrometheusService:
    def __init__(self):
        self.prom_url = 'http://host.docker.internal:9090'

    def query(self, query):
        response = requests.get(f'{self.prom_url}/api/v1/query', params={'query': query})
        data = response.json()
        return data['data']['result'][0]['value'][1]
    
    def batch_query(self, queries: List[str]) -> Dict[str, float]:
        results = {}
        for query in queries:
            try:
                result = self.query(query)
                # Extract the metric name or use a custom identifier for each query
                identifier = self.extract_metric_identifier(query)
                results[identifier] = float(result)  # Ensure the result is converted to a float
                logger.info(f"Fetched value {result} for query identifier {identifier}")
            except Exception as e:
                logger.error(f"Error while fetching value for query {query}: {e}")
        return results
    
    
    def batch_query(self, query_tuples: List[tuple]) -> Dict[str, float]:
        results = {}
        for query, func_name in query_tuples:
            try:
                result = self.query(query)
                identifier = self.extract_metric_identifier(func_name)
                results[identifier] = float(result)  # Ensure the result is converted to a float
            except Exception as e:
                logger.error(f"Error while fetching value for query {query}: {e}")
        return results
    



    def extract_metric_identifier(self, func_name: str) -> str:
        # Find the metric whose query function matches the input query
        for metric, query_func in METRIC_TO_QUERY_MAPPING.items():
            if query_func.__name__ == func_name:
                return metric
        return "unknown_metric"





