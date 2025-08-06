# Standard library imports for potential config loading
import json
from typing import Dict, Any

# Relative imports from subfiles
from .defaults import DefaultConfig  # Hyperparameters and simulation defaults
from .physics_constants import PhysicalConstants  # SI physical constants and derived properties

# Define __all__ to specify public exports
__all__ = [
    'DefaultConfig',
    'PhysicalConstants',
    'load_config_from_dict',  # Utility function defined below
]

def load_config_from_dict(config_dict: Dict[str, Any], config_class: type = DefaultConfig) -> Any:
    """
    Loads a configuration instance from a dictionary.
    This utility creates an instance of the specified config class (e.g., DefaultConfig or PhysicalConstants)
    by overriding default values with those provided in the dictionary. Useful for loading from JSON files
    or external sources, ensuring type safety and validation.

    Args:
        config_dict (Dict[str, Any]): Dictionary with key-value pairs matching config fields.
        config_class (type): The dataclass type to instantiate (default: DefaultConfig).

    Returns:
        Any: Instance of config_class with overridden values.

    Raises:
        ValueError: If dict keys do not match config fields or types mismatch.

    Note: Inspired by configurable parameters in Claude's artifacts; performs basic validation
          by checking field existence and attempting type coercion (e.g., str to float).
    """
    # Get default instance to use as base
    base_config = config_class()

    # Validate and override fields
    for key, value in config_dict.items():
        if not hasattr(base_config, key):
            raise ValueError(f"Invalid config key: {key}. Available fields: {list(base_config.__dataclass_fields__.keys())}")
        
        # Attempt type coercion if needed (e.g., str '1e-10' to float)
        field_type = type(getattr(base_config, key))
        try:
            coerced_value = field_type(value)
        except (TypeError, ValueError):
            raise ValueError(f"Type mismatch for {key}: expected {field_type}, got {type(value)}")
        
        setattr(base_config, key, coerced_value)

    return base_config

def load_config_from_json(file_path: str, config_class: type = DefaultConfig) -> Any:
    """
    Loads configuration from a JSON file into a config instance.
    Wrapper around load_config_from_dict for file-based loading.

    Args:
        file_path (str): Path to JSON file.
        config_class (type): Config class to use (default: DefaultConfig).

    Returns:
        Any: Config instance.

    Raises:
        FileNotFoundError: If file not found.
        json.JSONDecodeError: If invalid JSON.

    Note: Extension for user convenience; assumes flat JSON structure matching config fields.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return load_config_from_dict(config_dict, config_class)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in config file: {file_path}")
