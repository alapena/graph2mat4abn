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

import re

def read_fermi_level(path: str) -> float:
    """
    Return the last reported Fermi level (in eV) from a SIESTA aiida.out file.
    """
    pat = re.compile(r'Fermi\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
    fermi = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                fermi = float(m.group(1))
    if fermi is None:
        raise ValueError("Fermi level not found in file.")
    return fermi