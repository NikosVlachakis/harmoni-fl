import matplotlib.pyplot as plt
import json

# Load the JSON data
with open('../../all_data.json', 'r') as file:
    new_data = json.load(file)

# Analyzing the new data for total number of dropped out clients with and without the tool
dropped_out_with_tool = []
dropped_out_without_tool = []

for experiment in new_data:
    total_dropped_out = sum(entry['dropped_out'] for entry in experiment['data'])
    if experiment['experiment_description'] == 'tool-enabled':
        dropped_out_with_tool.append(total_dropped_out)
    else:  # 'without-tool'
        dropped_out_without_tool.append(total_dropped_out)


# Plotting the box plot for total number of dropped out clients
plt.figure(figsize=(10, 6))
plt.boxplot([dropped_out_without_tool, dropped_out_with_tool], labels=['Without Tool', 'With Tool'])
plt.title('Total Number of Dropped Out Clients With and Without Tool')
plt.ylabel('Total Dropped Out Clients')
plt.grid(True)

plt.savefig('../results/model_accuracy_experiment/dropped_out_comparison.pdf')

plt.show()
