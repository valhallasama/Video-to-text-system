"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and validate configuration from YAML files."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with path.open('r') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    @staticmethod
    def get_default_config_path() -> Path:
        """Get path to default configuration file."""
        return Path(__file__).resolve().parents[2] / "config" / "default.yaml"
