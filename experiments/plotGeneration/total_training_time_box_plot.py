import matplotlib.pyplot as plt
import json

data_path = "exp1_training_times_all_data"

# Load the JSON data
with open(f'../data/{data_path}.json', 'r') as file:
    data = json.load(file)

# Initialize lists to hold total training times
training_times_with_tool = []
training_times_without_tool = []

# Process each experiment
for experiment in data:
    # Find client1 data for the first and last rounds
    client1_data = [entry for entry in experiment['data'] if entry['container_name'] == 'client1']
    if client1_data:
        # Ensure data is sorted by round to correctly calculate total training time
        client1_data_sorted = sorted(client1_data, key=lambda x: int(x['server_round']))
        first_round = client1_data_sorted[0]
        last_round = client1_data_sorted[-1]
        
        # Calculate total training time in minutes
        total_training_time = (last_round['fit_end_time'] - first_round['fit_start_time']) / 60
        
        # Categorize by experiment description
        if 'tool-enabled' in experiment['experiment_description']:
            training_times_with_tool.append(total_training_time)
        else:
            training_times_without_tool.append(total_training_time)

# Plotting
plt.figure(figsize=(10, 6))
plt.boxplot([training_times_with_tool, training_times_without_tool], labels=['With Tool', 'Without Tool'])
plt.ylabel('Total Training Time (minutes)')
plt.title('Comparison of Total Training Time')

plt.savefig('../results/training_time_experiment/training_time_comparison.pdf')

plt.show()
