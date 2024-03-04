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


import matplotlib.pyplot as plt
import numpy as np

def plot_for_container(categorized_data, container_name, training_params):
    container_data = categorized_data.get(container_name, {})
    
    num_params = len(training_params)
    # Determine the grid layout: 2 rows. For columns, use min(3, num_params) to handle cases with fewer than 3 parameters.
    cols = min(3, num_params)
    rows = 2 if num_params > 3 else 1  # Use 2 rows if more than 3 parameters, else 1 row
    
    # Calculate figure size dynamically based on the number of subplots to ensure they are smaller and well-spaced
    fig_width = cols * 3  # Adjust width as needed
    fig_height = rows * 3  # Adjust height as needed
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, 
                            gridspec_kw={'hspace': 0.4, 'wspace': 0.5})  # Adjust spacing as needed
    
    # If only one parameter is plotted, axs is not an array, so we wrap it
    if num_params == 1:
        axs = np.array([[axs]])
    elif num_params <= 3:  # Make sure axs is 2D for uniform handling
        axs = np.array([axs])
    
    rounds = sorted(container_data.keys(), key=int)
    
    param_idx = 0  # Keep track of which parameter we're plotting
    for r in range(rows):
        for c in range(cols):
            if param_idx >= num_params:
                fig.delaxes(axs[r, c])  # Delete unused axes if any
                continue
            param = training_params[param_idx]
            data_to_plot = []
            for round in rounds:
                param_values = [entry[param] for entry in container_data[round] if param in entry]
                data_to_plot.append(param_values)
            
            # Create boxplot for each parameter
            axs[r, c].boxplot(data_to_plot, labels=rounds)
            axs[r, c].set_ylabel(param)
            
            param_idx += 1
    

    # Set common x-label for the last row of subplots or the only subplot
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlabel('Server Round')
    
    plt.xticks(np.arange(1, len(rounds) + 1), rounds)
    file_path = f'../results/{experiment_name}/{container_name}_params_adaptivity.pdf'
    plt.savefig(file_path)
    plt.show()


data_path = "exp1_adaptive_params_all_data"

# Load the JSON data
with open(f'../data/{data_path}.json', 'r') as file:
    data = json.load(file)

# Categorize the data
categorized_data = categorize_data(data)

# Specify the container name and parameters you want to plot
container_name = 'client2'
training_params = ['learning_rate', 'batch_size', 'epochs', 'freeze_layers_percentage', 'data_sample_percentage']
experiment_name = 'training_time_experiment'


# Plot
plot_for_container(categorized_data, container_name, training_params)