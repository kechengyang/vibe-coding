"""
Configuration Module for Employee Health Monitoring System.

This module handles application configuration and settings.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from ..data.models import UserSettings

logger = logging.getLogger("employee_health_monitor.utils.config")

class Config:
    """Class for handling application configuration."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "app": {
            "name": "Employee Health Monitor",
            "version": "0.1.0",
            "log_level": "INFO",
            "data_dir": "~/.employee_health_monitor",
            "db_filename": "health_data.db"
        },
        "camera": {
            "id": 0,
            "resolution": [640, 480],
            "fps": 15
        },
        "detection": {
            "posture": {
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "history_size": 20,
                "standing_threshold": 0.65,
                "sitting_threshold": 0.35
            },
            "drinking": {
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "history_size": 20,
                "hand_to_face_threshold": 0.15,
                "drinking_confidence_threshold": 0.55,
                "min_drinking_frames": 5
            }
        },
        "ui": {
            "theme": "light",
            "chart_colors": ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"],
            "refresh_interval": 1000  # milliseconds
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        # Determine config directory and file path
        if config_path is None:
            config_dir = Path.home() / ".employee_health_monitor"
            config_dir.mkdir(exist_ok=True)
            config_path = str(config_dir / "config.json")
        
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.user_settings = UserSettings()
        
        # Load configuration
        self.load()
        
        logger.info(f"Configuration initialized from {config_path}")
    
    def load(self) -> bool:
        """Load configuration from file.
        
        Returns:
            bool: True if configuration was loaded successfully
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update configuration with loaded values
                self._update_dict(self.config, loaded_config)
                
                # Load user settings
                if "user_settings" in loaded_config:
                    self.user_settings = UserSettings.from_dict(loaded_config["user_settings"])
                
                logger.info("Configuration loaded successfully")
                return True
            else:
                logger.info("No configuration file found, using defaults")
                return False
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def save(self) -> bool:
        """Save configuration to file.
        
        Returns:
            bool: True if configuration was saved successfully
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Prepare configuration for saving
            save_config = self.config.copy()
            save_config["user_settings"] = self.user_settings.to_dict()
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(save_config, f, indent=4)
            
            logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "app.name")
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Handle user settings separately
            if key_path.startswith("user_settings."):
                setting_name = key_path.split(".", 1)[1]
                return getattr(self.user_settings, setting_name, default)
            
            # Navigate through config dictionary
            parts = key_path.split(".")
            value = self.config
            
            for part in parts:
                if part in value:
                    value = value[part]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting configuration value for {key_path}: {str(e)}")
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set a configuration value by key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "app.name")
            value: Value to set
            
        Returns:
            bool: True if value was set successfully
        """
        try:
            # Handle user settings separately
            if key_path.startswith("user_settings."):
                setting_name = key_path.split(".", 1)[1]
                if hasattr(self.user_settings, setting_name):
                    setattr(self.user_settings, setting_name, value)
                    return True
                else:
                    logger.warning(f"Unknown user setting: {setting_name}")
                    return False
            
            # Navigate through config dictionary
            parts = key_path.split(".")
            target = self.config
            
            # Navigate to the parent of the target key
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            
            # Set the value
            target[parts[-1]] = value
            
            logger.info(f"Configuration value set: {key_path} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value for {key_path}: {str(e)}")
            return False
    
    def get_data_dir(self) -> str:
        """Get the data directory path.
        
        Returns:
            str: Absolute path to the data directory
        """
        data_dir = self.get("app.data_dir")
        
        # Expand ~ to user's home directory
        if data_dir.startswith("~"):
            data_dir = os.path.expanduser(data_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        return data_dir
    
    def get_db_path(self) -> str:
        """Get the database file path.
        
        Returns:
            str: Absolute path to the database file
        """
        data_dir = self.get_data_dir()
        db_filename = self.get("app.db_filename")
        
        return os.path.join(data_dir, db_filename)
    
    def get_user_settings(self) -> UserSettings:
        """Get user settings.
        
        Returns:
            UserSettings: User settings object
        """
        return self.user_settings
    
    def set_user_settings(self, settings: UserSettings) -> None:
        """Set user settings.
        
        Args:
            settings: User settings object
        """
        self.user_settings = settings
    
    def _update_dict(self, target: Dict, source: Dict) -> None:
        """Recursively update a dictionary with values from another dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_dict(target[key], value)
            else:
                # Update or add value
                target[key] = value


# Global configuration instance
_config_instance = None

def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Config: Global configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config()
    
    return _config_instance


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get configuration
    config = get_config()
    
    # Get some configuration values
    app_name = config.get("app.name")
    camera_id = config.get("camera.id")
    
    print(f"App name: {app_name}")
    print(f"Camera ID: {camera_id}")
    
    # Set a configuration value
    config.set("ui.theme", "dark")
    
    # Get user settings
    user_settings = config.get_user_settings()
    print(f"Standing goal: {user_settings.standing_goal_minutes} minutes")
    
    # Save configuration
    config.save()
