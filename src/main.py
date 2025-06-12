#!/usr/bin/env python3
"""
Employee Health Monitoring System - Main Application Entry Point

This module initializes and runs the main application.
"""
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / ".employee_health_monitor" / "app.log", mode="a"),
    ],
)
logger = logging.getLogger("employee_health_monitor")

# Ensure the log directory exists
log_dir = Path.home() / ".employee_health_monitor"
log_dir.mkdir(exist_ok=True)

def main():
    """Main application entry point."""
    try:
        logger.info("Starting Employee Health Monitoring System")
        
        # Import UI components here to avoid circular imports
        from src.ui.main_window import MainWindow
        
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
