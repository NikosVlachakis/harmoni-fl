import matplotlib.pyplot as plt
import json

from numpy import median

file_name = "exp2_acc_do_all_data"

# Load the JSON data
with open(f'../data/{file_name}.json', 'r') as file:
    new_data = json.load(file)

# Re-initialize dictionaries to hold accuracies for each round
accuracies_with_tool_new = {i: [] for i in range(1, 101)}
accuracies_without_tool_new = {i: [] for i in range(1, 101)}

# Process each experiment in the new data
for experiment in new_data:
    for entry in experiment['data']:
        round_number = int(entry['server_round'])
        accuracy = entry['aggregated_accuracy']
        
        if round_number <= 100:  # Skip rounds greater than 100
            if experiment['experiment_description'] == 'tool-enabled':
                accuracies_with_tool_new[round_number].append(accuracy)
            elif experiment['experiment_description'] == 'without-tool':
                accuracies_without_tool_new[round_number].append(accuracy)

# Calculate medians for each round in the new data
medians_with_tool_new = [median(accuracies) if accuracies else 0 for accuracies in accuracies_with_tool_new.values()]
medians_without_tool_new = [median(accuracies) if accuracies else 0 for accuracies in accuracies_without_tool_new.values()]

# Plotting for the new data
plt.figure(figsize=(12, 8))
plt.plot(range(1, 101), medians_with_tool_new, label='With Tool', marker='o', linestyle='-', markersize=5)
plt.plot(range(1, 101), medians_without_tool_new, label='Without Tool', marker='x', linestyle='-', markersize=5)
# plt.title('Model Accuracy Over Server Rounds')
plt.xlabel('Server Round')
plt.ylabel('Model Accuracy')
plt.legend()
plt.grid(True)

plt.savefig('../results/model_accuracy_experiment/accuracies_comparison_v2.pdf')

plt.show()

