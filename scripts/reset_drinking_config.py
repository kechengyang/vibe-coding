#!/usr/bin/env python
"""
Reset drinking detection configuration to use new threshold values.

This script resets the configuration file to use the updated default values,
particularly the lower drinking detection threshold.
"""
import sys
import os
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.utils.config import Config, get_config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reset_drinking_config")

def main():
    """Reset configuration to defaults."""
    try:
        # Get configuration path
        config = get_config()
        config_path = config.config_path
        
        # Print current values
        current_threshold = config.get("detection.drinking.drinking_confidence_threshold")
        logger.info(f"Current drinking threshold: {current_threshold}")
        
        # Create a new config instance (will use defaults)
        logger.info(f"Removing existing config file: {config_path}")
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # Create a new config with defaults
        new_config = Config()
        
        # Verify the new threshold
        new_threshold = new_config.get("detection.drinking.drinking_confidence_threshold")
        logger.info(f"New drinking threshold: {new_threshold}")
        
        # Save the new config
        new_config.save()
        logger.info(f"Configuration reset to defaults and saved to {config_path}")
        
        # Force-set the drinking threshold to ensure it's correct
        new_config.set("detection.drinking.drinking_confidence_threshold", 0.15)
        new_config.save()
        
        # Verify once more
        final_threshold = new_config.get("detection.drinking.drinking_confidence_threshold")
        logger.info(f"Final drinking threshold: {final_threshold}")
        
        return 0
    except Exception as e:
        logger.error(f"Error resetting configuration: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
