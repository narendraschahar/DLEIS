# config/__init__.py
```python
import os
import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model_config(model_name):
    """Get model-specific configuration"""
    config_dir = Path(__file__).parent / 'model_configs'
    config_path = config_dir / f'{model_name.lower()}_config.yaml'
    
    if not config_path.exists():
        raise ValueError(f"Configuration not found for model: {model_name}")
    
    return load_config(config_path)

def get_default_config():
    """Get default configuration"""
    config_path = Path(__file__).parent / 'default_config.yaml'
    return load_config(config_path)

__all__ = [
    'load_config',
    'get_model_config',
    'get_default_config'
]
```
