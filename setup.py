#!/usr/bin/env python3
"""
Setup script for Employee Health Monitoring System.
"""
import os
from setuptools import setup, find_packages

# Get the long description from the README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="employee-health-monitor",
    version="0.1.0",
    description="A desktop application for monitoring employee health behaviors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Employee Health Monitoring System Contributors",
    author_email="example@example.com",
    url="https://github.com/your-organization/employee-health-monitoring",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.7.0,<5.0.0",
        "mediapipe>=0.10.0,<0.11.0",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "PyQt6>=6.5.0,<7.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "plotly>=5.14.0,<6.0.0",
        "ultralytics>=8.0.0,<9.0.0",
        "scikit-learn>=1.2.0,<2.0.0",
        "onnxruntime>=1.14.0,<2.0.0",
        "pydantic>=1.10.7,<2.0.0",
        "apscheduler>=3.10.1,<4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "employee-health-monitor=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
