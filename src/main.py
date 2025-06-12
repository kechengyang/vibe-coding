#!/usr/bin/env python3
"""
Employee Health Monitoring System - Main Application Entry Point

This module initializes and runs the main application.
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import after path setup
from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.ui.main_window import MainWindow

# Set up logger
logger = setup_logger("employee_health_monitor")

def main():
    """Main application entry point."""
    try:
        logger.info("Starting Employee Health Monitoring System")
        
        # Load configuration
        config = get_config()
        
        # Create and show the main application window
        app = MainWindow(sys.argv)
        exit_code = app.exec()
        
        logger.info("Application exited with code: %d", exit_code)
        return exit_code
        
    except Exception as e:
        logger.exception("Unhandled exception in main application: %s", str(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())
