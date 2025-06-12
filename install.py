#!/usr/bin/env python3
"""
Installation helper for Employee Health Monitoring System.
Checks system requirements and installs dependencies.
"""
import os
import platform
import subprocess
import sys

def check_python_version():
    """Verify Python version meets requirements."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    print(f"✓ Python version {current_version[0]}.{current_version[1]} meets requirements.")

def check_camera_access():
    """Check if camera is accessible."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Warning: Could not access camera. Please check permissions.")
        else:
            ret, frame = cap.read()
            if ret:
                print("✓ Camera access verified.")
            else:
                print("Warning: Camera connected but could not capture frame.")
        cap.release()
    except ImportError:
        print("OpenCV not installed yet. Camera access will be checked after installation.")

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created.")
    else:
        print("✓ Virtual environment already exists.")

def install_requirements():
    """Install required packages from requirements.txt."""
    # Determine the pip executable path based on the virtual environment
    if platform.system() == "Windows":
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:
        pip_path = os.path.join("venv", "bin", "pip")
    
    print("Installing requirements...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    print("✓ Requirements installed successfully.")

def create_project_structure():
    """Create the basic project directory structure."""
    directories = [
        "src",
        "src/vision",
        "src/data",
        "src/ui",
        "src/ui/resources",
        "src/utils",
        "tests",
        "docs",
    ]
    
    print("Creating project structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create __init__.py files to make directories importable packages
    for directory in directories:
        if directory.startswith("src"):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write("# Package initialization\n")
    
    print("✓ Project structure created.")

def main():
    """Main installation function."""
    print("Employee Health Monitoring System - Installation")
    print("=" * 50)
    
    check_python_version()
    create_virtual_environment()
    install_requirements()
    create_project_structure()
    check_camera_access()
    
    print("\nInstallation completed successfully!")
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    
    print("\nTo start the application (once implemented):")
    print("    python -m src.main")

if __name__ == "__main__":
    main()
