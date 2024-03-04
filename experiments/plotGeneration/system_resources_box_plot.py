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
                'average_memory_usage_percentage': float(record['average_memory_usage_percentage']) if record['average_memory_usage_percentage'] != None else 0.0,
                'cpu_usase_percentage': float(record['cpu_usase_percentage']) if record['cpu_usase_percentage'] != None else 0.0
            })
    return categorized_data


import matplotlib.pyplot as plt
import numpy as np

def plot_for_container(categorized_data, container_name, training_params):
    container_data = categorized_data.get(container_name, {})
    
    num_params = len(training_params)
    # Determine the grid layout: 2 rows. For columns, use min(3, num_params) to handle cases with fewer than 3 parameters.
    cols = 2
    rows = 1 
    
    # Calculate figure size dynamically based on the number of subplots to ensure they are smaller and well-spaced
    fig_width = cols * 3.5  # Adjust width as needed
    fig_height = rows * 3.5 # Adjust height as needed
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, 
                            gridspec_kw={'hspace': 1, 'wspace': 0.5})  # Adjust spacing as needed
    
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
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin
    file_path = f'../results/training_time_experiment/{container_name}_system_resources_adaptivity.pdf'
    plt.savefig(file_path)
    plt.show()



# Load the JSON data
with open('../../all_data.json', 'r') as file:
    data = json.load(file)

# Categorize the data
categorized_data = categorize_data(data)

# Specify the container name and parameters you want to plot
container_name = 'client4'
training_params = ['average_memory_usage_percentage', 'cpu_usase_percentage']

# Plot
plot_for_container(categorized_data, container_name, training_params)