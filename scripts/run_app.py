#!/usr/bin/env python3
"""
Script to run the Employee Health Monitoring System.

This script provides a simple way to run the application.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Employee Health Monitoring System."""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check if virtual environment exists
    venv_dir = os.path.join(project_root, "venv")
    if not os.path.exists(venv_dir):
        print("Virtual environment not found. Running installation script...")
        install_script = os.path.join(project_root, "install.py")
        subprocess.run([sys.executable, install_script], check=True)
    
    # Determine the Python executable to use
    if os.name == "nt":  # Windows
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Unix-like
        python_executable = os.path.join(venv_dir, "bin", "python")
    
    # Check if the Python executable exists
    if not os.path.exists(python_executable):
        print(f"Python executable not found at {python_executable}")
        print("Please run the installation script manually:")
        print(f"python {os.path.join(project_root, 'install.py')}")
        return 1
    
    # Run the application
    print("Starting Employee Health Monitoring System...")
    main_module = os.path.join(project_root, "src", "main.py")
    result = subprocess.run([python_executable, main_module], check=True)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
