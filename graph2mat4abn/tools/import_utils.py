import importlib
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_object_from_module(class_name, module):
    try:
        return getattr(importlib.import_module(module), class_name)
    except AttributeError:
        return None  # Or raise an error if you prefer
    
def save_to_yaml(config, path):
    with open(path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)