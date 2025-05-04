import yaml
import os
from pathlib import Path

def load_config(config_path=None):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config file. If None, will look for config.yaml in the current directory
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            'config.yaml',
            'configs/config.yaml',
            '../config.yaml',
            Path(__file__).parent / 'config.yaml',
            Path(__file__).parent.parent / 'config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            raise FileNotFoundError("Config file not found in standard locations")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config