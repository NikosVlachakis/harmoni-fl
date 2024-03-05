import json
import matplotlib.pyplot as plt
import numpy as np

def categorize_data(data):
    categorized_data = {}
    for experiment in data:
        for record in experiment['data']:
            container_name = record['container_name']
            server_round = record['server_round']
            if container_name not in categorized_data:
                categorized_data[container_name] = {}
            if server_round not in categorized_data[container_name]:
                categorized_data[container_name][server_round] = []
            categorized_data[container_name][server_round].append({
                'average_memory_usage_percentage': float(record.get('average_memory_usage_percentage') or 0),
                'cpu_usase_percentage': float(record.get('cpu_usase_percentage') or 0)
            })
    return categorized_data

print()

def plot_resource_usage(categorized_data, container_name, resource_params):
    container_data = categorized_data.get(container_name, {})
    rounds = sorted(container_data.keys(), key=int)
    
    selected_rounds = [round for round in rounds if int(round) == 1 or int(round) % 5 == 0]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjusted for 1 row, 2 columns
    
    for idx, param in enumerate(resource_params):
        data_for_plotting = []
        for round in selected_rounds:
            if round in container_data:
                round_data = [entry[param] for entry in container_data[round] if param in entry]
                data_for_plotting.append(round_data)
        
        axs[idx].boxplot(data_for_plotting, positions=range(1, len(selected_rounds) + 1))
        axs[idx].set_title(param)
        axs[idx].grid(False)  # Grid removed
        
        axs[idx].set_xticks(range(1, len(selected_rounds) + 1))
        axs[idx].set_xticklabels(selected_rounds, rotation=45, ha="right")
        for label_idx, label in enumerate(axs[idx].get_xticklabels()):
            if label_idx % 2 == 0:
                label.set_visible(False)
    
    plt.tight_layout()
    file_path = f'../results/model_accuracy_experiment/{container_name}_system_resources_adaptivity.pdf'
    plt.savefig(file_path)
    plt.show()

file_name = "exp2_adaptive_params_all_data"

# Adapt the file path and container name as necessary
with open(f'../data/{file_name}.json', 'r') as file:
    data = json.load(file)

# Categorize the data
categorized_data = categorize_data(data)

# Specify the container name and parameters you want to plot
container_name = 'client5'  # Update as necessary
resource_params = ['average_memory_usage_percentage', 'cpu_usase_percentage']
experiment_name = 'resource_usage_experiment'

# Plotting with the adjusted script
plot_resource_usage(categorized_data, container_name, resource_params)
