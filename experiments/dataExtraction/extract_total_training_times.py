import os
import mlflow
import json


target_experiment_ids = [
"732595325010854792",
"274165294121367639",
"121812839154561502",
"858739433833093135",
"841870164085218879",
"397941700082241345",
"572069276148472884",
"895642181706698254",
"719385076820502398",
"664650711109893045"
]

def extract_mlflow_data_based_on_experiment(experiment_id):
    
    runs = mlflow.search_runs([experiment_id])
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
        if run.get('tags.mlflow.runName') == 'Experiment Details' or not run['tags.container_name'].startswith('client'):
            continue
        
        run_data = {
            'server_round': run['tags.server_round'],
            'container_name': run['tags.container_name'],
            'fit_duration': run['metrics.fit_duration'],
            'fit_start_time': run['metrics.fit_start_time'],
            'fit_end_time': run['metrics.fit_end_time'],
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
                    if experiment_data['data'] and experiment_data['experiment_name'].startswith('v1'):
                        all_experiments_data.append(experiment_data)
            except Exception as e:
                print(f"Error processing experiment {experiment_id}: {str(e)}")
                continue  # Skip to the next experiment if an error occurs
    
    return all_experiments_data

# Adjust the mlruns_path if your MLflow data is stored in a different location
all_data = extract_all_experiments_data('mlruns')

# save all data to a file
with open('exp1_training_times_all_data.json', 'w') as f:
    json.dump(all_data, f)
