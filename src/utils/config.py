"""
Configuration management for the Video Analytics Platform.
Handles loading config from files and environment variables.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

class Config:
    """
    Configuration manager that handles loading and accessing
    application settings from files and environment variables.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Path to configuration directory (default: PROJECT_ROOT/config)
        """
        # Determine the project root and config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Find project root (2 levels up from this file)
            file_dir = Path(__file__).resolve().parent
            project_root = file_dir.parent.parent
            self.config_dir = project_root / "config"
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
        
        # Initialize config with default values
        self._config = self._get_default_config()
        
        # Environment name (default to development)
        self.env = os.environ.get("APP_ENV", "development")
        
        # Load configuration
        self.reload()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration values.
        
        Returns:
            Dictionary of default configuration settings
        """
        return {
            # API settings
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            },
            
            # Video ingestion settings
            "video_ingestion": {
                "default_resolution": [640, 480],
                "default_framerate": 30,
                "buffer_size": 10,
                "gstreamer_path": "gst-launch-1.0"
            },
            
            # Analytics settings
            "analytics": {
                "enabled": True,
                "default_model": "yolov5s",
                "confidence_threshold": 0.5,
                "heatmap_decay": 0.95
            },
            
            # Streaming settings
            "streaming": {
                "base_port": 5000,
                "rtsp_enabled": True,
                "hls_enabled": False,
                "udp_enabled": True,
                "bitrate": 800
            },
            
            # MQTT settings
            "mqtt": {
                "enabled": False,
                "broker": "localhost",
                "port": 1883,
                "topic_prefix": "video_analytics"
            },
            
            # Logging settings
            "logging": {
                "level": "INFO",
                "file_enabled": False,
                "file_path": "logs/app.log"
            },
            
            # Storage settings
            "storage": {
                "recording_enabled": False,
                "recording_path": "recordings",
                "max_disk_usage_gb": 10
            },
            
            # Monitoring settings
            "monitoring": {
                "enabled": False,
                "influxdb_url": "http://localhost:8086",
                "influxdb_token": "",
                "influxdb_org": "video_analytics",
                "influxdb_bucket": "metrics"
            }
        }
    
    def reload(self) -> None:
        """Reload configuration from files and environment variables."""
        # Start with default config
        config = self._get_default_config()
        
        # Load default configuration file if it exists
        default_config_path = self.config_dir / "default.json"
        if default_config_path.exists():
            try:
                with open(default_config_path, "r") as f:
                    default_config = json.load(f)
                    self._deep_update(config, default_config)
                logger.info(f"Loaded default configuration from {default_config_path}")
            except Exception as e:
                logger.error(f"Error loading default config: {e}")
        
        # Load environment-specific configuration file if it exists
        env_config_path = self.config_dir / f"{self.env}.json"
        if env_config_path.exists():
            try:
                with open(env_config_path, "r") as f:
                    env_config = json.load(f)
                    self._deep_update(config, env_config)
                logger.info(f"Loaded {self.env} configuration from {env_config_path}")
            except Exception as e:
                logger.error(f"Error loading {self.env} config: {e}")
        
        # Override with environment variables
        self._apply_env_overrides(config)
        
        # Save the loaded configuration
        self._config = config
        logger.info(f"Configuration loaded successfully for environment: {self.env}")
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._deep_update(target[key], value)
            else:
                # Update or add values
                target[key] = value
    
    def _apply_env_overrides(self, config: Dict) -> None:
        """
        Override configuration with environment variables.
        
        Environment variables should be in the format:
        APP_SECTION_KEY=value (e.g., APP_API_PORT=8080)
        
        Args:
            config: Configuration dictionary to update
        """
        prefix = "APP_"
        for env_name, env_value in os.environ.items():
            if env_name.startswith(prefix):
                # Remove prefix and split into sections
                parts = env_name[len(prefix):].lower().split("_")
                
                if len(parts) < 2:
                    continue
                
                # Navigate to the right section of the config
                current = config
                for section in parts[:-1]:
                    if section not in current:
                        current[section] = {}
                    current = current[section]
                
                # Set the value, converting to appropriate type
                key = parts[-1]
                original_value = current.get(key)
                
                if isinstance(original_value, bool):
                    # Convert string to boolean
                    current[key] = env_value.lower() in ("true", "yes", "1", "y")
                elif isinstance(original_value, int):
                    # Convert string to integer
                    try:
                        current[key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Could not convert {env_name}={env_value} to int")
                elif isinstance(original_value, float):
                    # Convert string to float
                    try:
                        current[key] = float(env_value)
                    except ValueError:
                        logger.warning(f"Could not convert {env_name}={env_value} to float")
                elif isinstance(original_value, list):
                    # Convert comma-separated string to list
                    current[key] = [item.strip() for item in env_value.split(",")]
                else:
                    # Use as string
                    current[key] = env_value
                
                logger.debug(f"Config override from environment: {env_name}={env_value}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if section/key doesn't exist
            
        Returns:
            Configuration value or default
        """
        if section in self._config and key in self._config[section]:
            return self._config[section][key]
        return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Configuration section name
            
        Returns:
            Dictionary containing the section or empty dict if not found
        """
        return self._config.get(section, {})
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()


# Create singleton instance
_config_instance: Optional[Config] = None

def get_config() -> Config:
    """
    Get or create the config instance.
    
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance