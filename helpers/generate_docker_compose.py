def create_docker_compose(num_clients):
    docker_compose_content = f"""
version: '3'
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - 9090:9090
    mem_limit: 300m
    command:
      - --config.file=/etc/prometheus/prometheus.yml
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - cadvisor
    restart: always

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: cadvisor
    restart: always
    privileged: true
    mem_limit: 150m
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
    mem_limit: 300m
    restart: always
    volumes:
      - grafana-storage:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
      - cadvisor

  server:
    container_name: server
    shm_size: '6g'
    build:
      context: .
      dockerfile: Dockerfile
    command: python server.py
    restart: always
    environment:
      CLIENTS: "client1:6001,client2:6002,client3:6003"
      FLASK_RUN_PORT: 6000
      DOCKER_HOST_IP: host.docker.internal
      MLFLOW_TRACKING_URI: http://mlflow_server:5010
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock      
    ports:
      - "6000:6000"
      - "8265:8265"
    depends_on:
      - mlflow_server
      - prometheus
      - grafana

  mlflow_server:
    container_name: mlflow_server
    restart: always
    build:
      context: .
      dockerfile: Dockerfile
    mem_limit: 500m
    command: mlflow server --host 0.0.0.0 --port 5010 --default-artifact-root /mlflow/artifacts --backend-store-uri /mlflow/runs
    volumes:
      - ./mlflow:/mlflow
      - ./mlruns:/mlruns 
    ports:
      - "5010:5010"
"""
    # Add client services
    for i in range(1, num_clients + 1):
        docker_compose_content += f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: Dockerfile
    command: python client.py --server_address=server:8080
    restart: always
    mem_limit: 4g
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
      MLFLOW_TRACKING_URI: http://mlflow_server:5010
"""

    docker_compose_content += "volumes:\n  grafana-storage:\n"

    with open('docker-compose.yml', 'w') as file:
        file.write(docker_compose_content)

if __name__ == "__main__":
    num_clients = 5  # You can change this number as needed
    create_docker_compose(num_clients)
