#!/usr/bin/env python3
"""
Employee Health Monitoring System - Main Application Entry Point

This module initializes and runs the main application.
"""
import sys
import os
import ssl
import urllib.request
from pathlib import Path

# Fix for SSL certificate verification error on macOS
# This is needed for MediaPipe to download models on first run
def fix_ssl_certificates():
    """Bypass SSL certificate verification for model downloads."""
    # Bypass SSL verification (not recommended for production, but necessary for model downloads)
    ssl._create_default_https_context = ssl._create_unverified_context

# Apply SSL fix before any imports that might trigger model downloads
fix_ssl_certificates()

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
