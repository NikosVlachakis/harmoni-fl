import requests
import json
from services.prometheus_queries import *
import logging

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class PrometheusService:
    def __init__(self):
        self.prom_url = 'http://host.docker.internal:9090'

    def query(self, query):
        response = requests.get(f'{self.prom_url}/api/v1/query', params={'query': query})
        data = response.json()
        return data['data']['result'][0]['value'][1]

    def container_specific_avg_cpu_usage(self, container_name: str, start_timestamp: int, end_timestamp: int):
        query = container_specific_avg_cpu_usage_query(container_name, start_timestamp, end_timestamp)  # Use the query function
        result = self.query(query)
        return result

    def container_specific_max_cpu_usage(self, container_name: str, start_timestamp: int, end_timestamp: int):
        query = container_specific_max_cpu_usage_query(container_name, start_timestamp, end_timestamp)
        result = self.query(query)
        return result
    
    def container_specific_max_memory_usage(self, container_name: str, start_timestamp: int, end_timestamp: int):
        query = container_specific_max_memory_usage_query(container_name, start_timestamp, end_timestamp)
        result = self.query(query)
        return result

