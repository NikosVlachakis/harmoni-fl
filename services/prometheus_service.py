import requests
import json
from services.prometheus_queries import *
import logging
from typing import Dict, List
from helpers.mappings import *

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class PrometheusService:
    def __init__(self):
        self.prom_url = 'http://prometheus:9090'
        
    def query(self, query):
        response = requests.get(
            f'{self.prom_url}/api/v1/query',
            params={'query': query}
        )
        data = response.json()

        # Process the response
        if response.status_code == 200 and 'data' in data and 'result' in data['data']:
            results = data['data']['result']
            if results:
                # Assuming the result contains a single series with a single value
                average = float(results[0]['value'][1])  # Extract the value
                return average
            else:
                logger.info("No data returned for the query.")
                return 0
        else:
            error_message = data.get('error', 'Unknown error')
            logger.error(f"Error while querying Prometheus: {error_message}")
            raise Exception(f"Error while querying Prometheus: {error_message}")

    
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
    






