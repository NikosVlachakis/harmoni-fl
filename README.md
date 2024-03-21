# HarmoniFL: Resource-Adaptive Federated Learning Tool

Detailed information about the theoretical aspects, framework, and methodologies underpinning the HarmoniFL tool can be found in the accompanying theoretical paper. For a comprehensive understanding of the development process that led to HarmoniFL, please refer to [the theoretical paper](https://example.com/theoretical-paper).

## Abstract

This thesis introduces HarmoniFL, an initiative engineered to navigate the complexities of resource heterogeneity within federated learning frameworks. Focused on enhancing efficiency and inclusivity, HarmoniFL employs a dynamic client selection protocol, leveraging real-time metrics like CPU load, memory availability, and network bandwidth to optimize device participation in learning tasks. Adaptive strategies—ranging from data sampling adjustments to epoch reduction and batch size optimization for high-demand devices—address the challenges of resource diversity. Our experimental analysis focused on two primary goals: minimizing training duration for less capable devices and improving the accuracy of the aggregated global model. Results demonstrate that HarmoniFL effectively reduces training times and enhances model performance, underscoring its potential to foster more equitable device participation in federated learning tasks without sacrificing learning quality.

## Dynamic Training Configuration

The training process for each client is designed to be dynamic, allowing for adjustments based on the available computational resources. This ensures both optimal performance and efficient resource utilization across different hardware environments. 

### Configuration via `criteria.yaml`

To customize the training parameters dynamically, settings are specified in the `criteria.yaml` file within the `config` folder. This configuration file is pivotal in adjusting training parameters like batch size, learning rate, and more, tailored to the real-time capabilities of each client's hardware.

### Key Elements of `criteria.yaml`

The `criteria.yaml` file contains several criteria, each defined with specific types and configurations. These criteria are used to dynamically adjust training parameters based on hardware utilization metrics such as CPU and memory usage, as well as network bandwidth. Each criterion is structured as follows:

- `type`: Identifies the adjustment strategy, e.g., adaptive batch size, learning rate adjustment, etc.
- `blocking`: Indicates if the training should be halted when this criterion is not met.
- `active`: Specifies whether the criterion is currently enabled or disabled.
- `config`: Contains the configuration settings for the criterion, including thresholds and adjustment factors.

### Examples of Criteria

1. **Adaptive Learning Rate**: Adjusts the learning rate based on CPU utilization, helping to balance computational load and training efficiency.
2. **Adaptive Batch Size**: Modifies the batch size in response to memory utilization, optimizing memory usage without compromising training effectiveness.
3. **Model Layer Freezing**: Reduces computational complexity by freezing model layers when CPU utilization is high, ensuring stable training performance.



## Running the System

Docker must be installed and the Docker daemon running on your server. If you don't already have Docker installed, you can get [installation instructions for your specific Linux distribution or macOS from Docker](https://docs.docker.com/engine/install/). Besides Docker, the only extra requirement is having Python installed. You don't need to create a new environment for this example since all dependencies will be installed inside Docker containers automatically.

### Step 1: Configure Docker Compose

Execute the following command to run the `helpers/generate_docker_compose.py` script. This script creates the docker-compose configuration needed to set up the environment.

```bash
python helpers/generate_docker_compose.py
```

Within the script, specify the number of clients (`total_clients`) and resource limitations for each client in the `client_configs` array. You can adjust the number of rounds by passing `--num_rounds` to the above command.

### Step 2: Build and Launch Containers

1. **Execute Initialization Script**:

   - To build the Docker images and start the containers, use the following command:

     ```bash
     docker-compose up
     ```

2. **Services Startup**:

   - Several services will automatically launch as defined in your `docker-compose.yml` file:

     - **Monitoring Services**: Prometheus for metrics collection, Cadvisor for container monitoring, and Grafana for data visualization.
     - **Flower Federated Learning Environment**: The Flower server and client containers are initialized and start running.

3. **Automated Grafana Configuration**:

   Grafana is configured to load pre-defined data sources and dashboards for immediate monitoring, facilitated by provisioning files. The provisioning files include `prometheus-datasource.yml` for data sources, located in the `./config/provisioning/datasources` directory, and `dashboard_index.json` for dashboards, in the `./config/provisioning/dashboards` directory. The `grafana.ini` file is also tailored to enhance user experience:
     - **Admin Credentials**: We provide default admin credentials in the `grafana.ini` configuration, which simplifies access by eliminating the need for users to go through the initial login process.
     - **Default Dashboard Path**: A default dashboard path is set in `grafana.ini` to ensure that the dashboard with all the necessary panels is rendered when Grafana is accessed.

   Visit `http://localhost:3000` to enter Grafana, where the automated setup greets you with a series of pre-configured dashboards, ready for immediate monitoring and customizable to fit specific requirements. The `dashboard_index.json` file, pivotal for dashboard configuration, outlines panel structures and settings for key metrics like model accuracy and CPU usage. This setup, enhanced by direct volume mappings in Docker Compose, ensures Grafana's readiness from startup without additional configuration, providing an insightful snapshot into the federated learning system's performance.

4. **Begin Training Process**:

   - The federated learning training automatically begins once all client containers are successfully connected to the Flower server. This synchronizes the learning process across all participating clients.


## Running Experiments

To conduct experiments and analyze results within the HarmoniFL framework, follow these steps:

1. **Data Extraction**: In the `experiments/dataExtraction` directory, you'll find scripts designed to extract data from MLflow. These scripts are essential for gathering the metrics and results generated during the federated learning process.

2. **Plot Generation**: After extracting the data, navigate to the `experiments/plotGeneration` folder. Here, scripts are available to create visual representations of the experiment results, such as accuracy over training rounds.

3. **Pre-configured Data and Results**: For your convenience, we've included pre-collected data and results from our experiments. This allows for a quick start in analyzing the performance and efficiency of the HarmoniFL system without the need for immediate experiment replication.

By following these steps, you can efficiently run experiments, extract necessary data, and generate plots to visualize the performance and outcomes of the HarmoniFL framework.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
