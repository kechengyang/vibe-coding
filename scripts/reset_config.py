#!/usr/bin/env python3
"""
Reset Configuration Script for Employee Health Monitoring System.

This script resets the configuration file to use the default values.
"""
import os
import sys
import logging

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import after path setup
from src.utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reset_config")

def main():
    """Reset the configuration file to use the default values."""
    try:
        # Get configuration
        config = get_config()
        
        # Get config path
        config_path = config.config_path
        
        # Check if config file exists
        if os.path.exists(config_path):
            # Backup existing config
            backup_path = f"{config_path}.bak"
            os.rename(config_path, backup_path)
            logger.info(f"Existing configuration backed up to {backup_path}")
        
        # Save default configuration
        success = config.save()
        
        if success:
            logger.info(f"Configuration reset to defaults at {config_path}")
            return 0
        else:
            logger.error("Failed to reset configuration")
            return 1
            
    except Exception as e:
        logger.exception(f"Error resetting configuration: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
