# Federated Learning Strategy Auditor Tool

## Overview

This project develops a tool designed to address device heterogeneity in federated learning environments. The tool features a dynamic client selection strategy that evaluates device capabilities based on metrics like CPU usage, network bandwidth, and memory. It leverages Prometheus, cadvisor, MLflow, and Grafana for metrics sourcing, tracking, and visualization.

## Features

- **Dynamic Client Selection:** Assesses device capabilities using Prometheus and cadvisor, enabling informed client participation in learning iterations.
- **Historical Data Analysis:** Queries past training data using PromQL for better decision-making in client selection.
- **Detailed Tracking and Visualization:** Uses MLflow and Grafana for monitoring and visualizing device performance and learning progress.
- **Adaptive Learning Strategies:** Implements various strategies like gradient sparsification and model weight's precision to manage device diversity.

## User Specifications

All user specifications are detailed in the `criteria.yaml` file, which is located under `PROJECTS_ROOT_FOLDER/config/criteria.yaml`. These include:

- **Leveraging Sparsification:** Applied to manage data transmission efficiently when network bandwidth is limited.
- **Selective Participation for Slower Devices:** Ensures devices with lower memory and CPU capabilities can participate effectively without being overwhelmed.
- **Adaptive Data Sampling:** Adjusts data sample size for training based on the device's RAM utilization, ensuring optimal memory use.
- **Learning Rate Adjustment:** Modifies learning rate dynamically in response to variations in network connectivity and speed.
- **Epoch Reduction for High CPU Utilization:** Tailors the number of training epochs for devices experiencing high CPU load to prevent overuse.
- **Adaptive Batch Size:** Alters training batch size according to the device's memory capacity to maintain efficient operations.
- **Model's Layer Freezing:** Adjusts the complexity of the model's layers, especially in fully connected sections, based on the device's CPU usage to ease computational demands.
- **Gradient Clipping:** Implements gradient size limitations during backpropagation for devices with constrained computational resources.
- **Weight Precision Adjustment:** Modifies the precision of the model's weights during training for devices experiencing high CPU utilization, optimizing computational efficiency.

## Setup and Installation

To set up the tool, navigate to the project's root folder and run `./docker_init.sh`. This script will execute the Docker Compose file, setting up each of the services as defined in the Docker Compose configuration.

## License

This project is licensed under the MIT License.
