import yaml

def load_criteria_config(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
