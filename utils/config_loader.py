"""
Configuration Loader Utility
Loads experiment configuration from YAML file and provides helper functions
"""

import yaml
import os

def load_config(config_path="configs/experiment.yaml"):
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_hyperparams(config):
    """
    Extract hyperparameters from config for easy access.
    
    Args:
        config (dict): Full configuration dictionary
        
    Returns:
        dict: Hyperparameters only
    """
    return {
        'seq_len': config['dataset']['seq_len'],
        'batch_size': config['training']['batch_size'],
        'epochs': config['training']['epochs'],
        'lr': config['training']['learning_rate'],
        'd_model': config['model']['d_model'],
        'n_layers': config['model']['n_layers'],
    }
