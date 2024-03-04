import os
import mlflow
import json


target_experiment_ids = ['690442299376853048','128706956441954763','236190549831905507']

def extract_mlflow_data_based_on_experiment(experiment_id):
    
    runs = mlflow.search_runs(experiment_ids=[experiment_id]) 
    experiment_description = None  # Variable to hold the experiment description when found
    experiment_name = None  # Variable to hold the experiment name when found

    # First loop to find the experiment description
    for _, run in runs.iterrows():
        if run.get('tags.mlflow.runName') == 'Experiment Details':
            experiment_description = run.get('params.experiment_description')
            experiment_name = run.get('params.experiment_name')
            break  # Exit the loop once the experiment description is found

    # Initialize the structure to hold experiment name and data
    experiment_data = {
        'experiment_description': experiment_description,
        'experiment_name': experiment_name,  
        'data': []
    }

    # Second loop to extract data from other runs
    for _, run in runs.iterrows():
        if run.get('tags.mlflow.runName') == 'Experiment Details' or not run['tags.container_name'].startswith('fl-server'):
            continue
        
        run_data = {
            'server_round': run['tags.server_round'],
            'dropped_out': run['metrics.dropped_out'],
            'aggregated_accuracy': run['metrics.aggregated_accuracy'],
        }

        experiment_data['data'].append(run_data)

    return experiment_data



def extract_all_experiments_data(mlruns_path='/mlruns'):
    all_experiments_data = []
    
    # List all experiment directories in the /mlruns folder
    for experiment_id in os.listdir(mlruns_path):
        experiment_path = os.path.join(mlruns_path, experiment_id)
        if os.path.isdir(experiment_path):
            try:
                if experiment_id in target_experiment_ids:
                    experiment_data = extract_mlflow_data_based_on_experiment(experiment_id)
                    all_experiments_data.append(experiment_data)
                    # if experiment_data['data'] and experiment_data['experiment_name'].startswith('v2'):
                        # all_experiments_data.append(experiment_data)
            except Exception as e:
                print(f"Error processing experiment {experiment_id}: {str(e)}")
                continue  # Skip to the next experiment if an error occurs
    
    return all_experiments_data

# Adjust the mlruns_path if your MLflow data is stored in a different location
all_data = extract_all_experiments_data('mlruns')

# save all data to a file
with open('all_data.json', 'w') as f:
    json.dump(all_data, f)


# Demonstration: print part of the extracted data for each experiment
# for experiment_data in all_data:
#     print(f"Experiment Name: {experiment_data['experiment_name']}")
#     print(f"Experiment Description: {experiment_data['experiment_description']}")
#     for item in experiment_data['data']:  # Print first 5 for brevity
#         print(item)
#     print("---------")