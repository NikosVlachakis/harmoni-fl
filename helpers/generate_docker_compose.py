import logging
import random
import argparse

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

parser = argparse.ArgumentParser(description='Generated Docker Compose')
parser.add_argument('--total_clients', type=int, default=2, help="Total clients to spawn (default: 2)")
parser.add_argument('--num_rounds', type=int, default=100, help="Number of FL rounds (default: 100)")
parser.add_argument('--random', type=bool, default=False, help='Randomize client configurations (default: False)')
parser.add_argument('--convergence_accuracy', type=float, default=0.8, help='Convergence accuracy (default: 0.8)')
parser.add_argument('--dpsgd', type=bool, default=False, help='DPSGD or not (default: False)')


def create_docker_compose(args):
    # cpus is used to set the number of CPUs available to the container as a fraction of the total number of CPUs on the host machine.
    # mem_limit is used to set the memory limit for the container.
    client_configs = [
        {'mem_limit': '3g',  "cpus": 4},
        # {'mem_limit': '4g',  "cpus": 3},
        # {'mem_limit': '5g',  "cpus": 2.5},
        {'mem_limit': '6g',  "cpus": 1}
        
        # Add or modify the configurations depending on your host machine
    ]

    docker_compose_content = f"""
version: '3'
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - 9090:9090
    mem_limit: 500m
    command:
      - --config.file=/etc/prometheus/prometheus.yml
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - cadvisor
    restart: on-failure

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: cadvisor
    restart: on-failure
    privileged: true
    mem_limit: 500m
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
      - /var/run/docker.sock:/var/run/docker.sock  

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - 3000:3000
    mem_limit: 400m
    restart: on-failure
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./config/grafana.ini:/etc/grafana/grafana.ini
      - ./config/provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./config/provisioning/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
      - cadvisor
    command:
      - --config=/etc/grafana/grafana.ini
      
  mlflow_server:
    container_name: mlflow_server
    restart: always
    build:
      context: .
      dockerfile: Dockerfile
    mem_limit: 500m
    command: mlflow server --host 0.0.0.0 --port 5010  --backend-store-uri /mlruns
    volumes:
      - .:/app
      - ./mlruns:/mlruns
      - ./mlflow:/mlflow
    ports:
      - "5010:5010"

  server:
    container_name: server
    shm_size: '6g'
    build:
      context: .
      dockerfile: Dockerfile
    command: python server.py --number_of_rounds={args.num_rounds} --convergence_accuracy={args.convergence_accuracy}
    environment:
      FLASK_RUN_PORT: 6000
      DOCKER_HOST_IP: host.docker.internal
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock      
    ports:
      - "6000:6000"
      - "8265:8265"
      - "8000:8000"
    depends_on:
      - prometheus
      - grafana
"""
    # Add client services
    for i in range(1, args.total_clients + 1):
        if args.random:
            config = random.choice(client_configs)
        else:
            config = client_configs[(i-1) % len(client_configs)]
        docker_compose_content += f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: Dockerfile
    command: python client.py --server_address=server:8080  --client_id={i} --total_clients={args.total_clients} --dpsgd={args.dpsgd}
    mem_limit: {config['mem_limit']}
    deploy:
      resources:
        limits:
          cpus: "{(config['cpus'])}"
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - "{6000 + i}:{6000 + i}"
    depends_on:
      - server
    environment:
      FLASK_RUN_PORT: {6000 + i}
      container_name: client{i}
      DOCKER_HOST_IP: host.docker.internal
"""

    docker_compose_content += "volumes:\n  grafana-storage:\n"

    with open('docker-compose.yml', 'w') as file:
        file.write(docker_compose_content)

if __name__ == "__main__":
    args = parser.parse_args()
    create_docker_compose(args)
