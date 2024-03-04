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
                'data_sample_percentage': float(record['data_sample_percentage']),
                'learning_rate': float(record['learning_rate']),
                'batch_size': int(record['batch_size']),
                'epochs': int(record['epochs']),
                'freeze_layers_percentage': int(record['freeze_layers_percentage'])
            })
    return categorized_data

def plot_all_params_with_corrected_labels(categorized_data, container_name, training_params):
    container_data = categorized_data.get(container_name, {})
    rounds = sorted(container_data.keys(), key=int)
    
    selected_rounds = [round for round in rounds if int(round) == 1 or int(round) % 5 == 0]
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    for idx, param in enumerate(training_params):
        if idx >= len(axs):
            break
        data_for_plotting = []
        for round in selected_rounds:
            if round in container_data:
                round_data = [entry[param] for entry in container_data[round] if param in entry]
                data_for_plotting.append(round_data)
        
        axs[idx].boxplot(data_for_plotting, positions=range(1, len(selected_rounds) + 1))
        axs[idx].set_title(param)
        axs[idx].grid(False)
        
        axs[idx].set_xticks(range(1, len(selected_rounds) + 1))
        axs[idx].set_xticklabels(selected_rounds, rotation=45, ha="right")
        for label_idx, label in enumerate(axs[idx].get_xticklabels()):
            if label_idx % 2 == 0:
                label.set_visible(False)
    
    for idx in range(len(training_params), len(axs)):
        fig.delaxes(axs[idx])
    
    plt.tight_layout()
    # plt.grid(False)
    file_path = f'../results/{experiment_name}/{container_name}_params_adaptivity.pdf'
    plt.savefig(file_path)
    plt.show()

# Load the JSON data
with open('../../all_data.json', 'r') as file:
    data = json.load(file)

# Categorize the data
categorized_data = categorize_data(data)

# Specify the container name and parameters you want to plot
container_name = 'client2'
training_params = ['learning_rate', 'batch_size', 'epochs', 'freeze_layers_percentage', 'data_sample_percentage']
experiment_name = 'model_accuracy_experiment'

# Plotting with improved label handling for the specified container
plot_all_params_with_corrected_labels(categorized_data, container_name, training_params)
