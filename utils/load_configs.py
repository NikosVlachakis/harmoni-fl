import json
import yaml


def load_strategy_config():
        with open('config/strategy_config.json') as f:
            config = json.load(f)
        return config


def load_criteria_config(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
