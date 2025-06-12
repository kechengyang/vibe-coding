# Employee Health Monitoring System - Developer Guide

This guide is intended for developers who want to understand, modify, or contribute to the Employee Health Monitoring System.

## Project Overview

The Employee Health Monitoring System is a desktop application that uses computer vision to monitor and promote healthy work habits. It detects when users stand up from their desk and when they drink water, and provides statistics and reminders to help them maintain good health habits.

## Architecture

The application follows a modular architecture with the following main components:

### Vision Module (`src/vision/`)

The vision module handles computer vision tasks:

- `capture.py`: Manages webcam access and frame processing
- `pose_detection.py`: Detects sitting and standing postures
- `action_detection.py`: Detects drinking actions

### Data Module (`src/data/`)

The data module handles data storage and analysis:

- `database.py`: Manages the SQLite database for storing events and settings
- `models.py`: Defines data structures and models
- `analytics.py`: Analyzes health data and generates statistics

### UI Module (`src/ui/`)

The UI module handles the user interface:

- `main_window.py`: Defines the main application window
- `dashboard.py`: Displays health statistics and visualizations
- `settings.py`: Allows users to configure the application

### Utils Module (`src/utils/`)

The utils module provides utility functions:

- `config.py`: Manages application configuration
- `logger.py`: Handles application logging
- `notifications.py`: Manages system notifications

## Technology Stack

- **Python**: Core programming language
- **OpenCV**: Computer vision library for image processing
- **MediaPipe**: Framework for building multimodal machine learning pipelines
- **PyQt6**: GUI framework for building the user interface
- **SQLite**: Embedded database for storing data
- **Matplotlib/Plotly**: Libraries for data visualization
- **NumPy/Pandas**: Libraries for data manipulation and analysis

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- A code editor (e.g., Visual Studio Code)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/your-organization/employee-health-monitoring.git
cd employee-health-monitoring
```

2. Run the installation script:
```bash
python install.py
```

3. Activate the virtual environment:
```bash
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

4. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

5. Set up pre-commit hooks:
```bash
pre-commit install
```

### Running the Application

```bash
python -m src.main
```

### Running Tests

```bash
pytest
```

## Code Organization

```
employee-health-monitoring/
├── .github/                      # GitHub workflows and templates
├── docs/                         # Documentation
├── src/                          # Source code
│   ├── vision/                   # Computer vision modules
│   │   ├── capture.py            # Video capture functionality
│   │   ├── pose_detection.py     # Standing detection
│   │   └── action_detection.py   # Drinking detection
│   ├── data/                     # Data handling
│   │   ├── database.py           # SQLite interface
│   │   ├── models.py             # Data models
│   │   └── analytics.py          # Data analysis
│   ├── ui/                       # User interface
│   │   ├── main_window.py        # Main application window
│   │   ├── dashboard.py          # Statistics dashboard
│   │   ├── settings.py           # User settings
│   │   └── resources/            # UI resources
│   └── utils/                    # Utility functions
│       ├── config.py             # Configuration handling
│       ├── logger.py             # Logging functionality
│       └── notifications.py      # Notification system
├── tests/                        # Test suite
│   ├── test_vision.py            # Vision module tests
│   ├── test_data.py              # Data handling tests
│   └── test_ui.py                # UI tests
├── scripts/                      # Utility scripts
│   ├── install.py                # Installation helper
│   └── benchmark.py              # Performance testing
├── .gitignore                    # Git ignore file
├── LICENSE                       # Open source license
├── README.md                     # Project documentation
├── requirements.txt              # Pip requirements file
├── requirements-dev.txt          # Development requirements
├── setup.py                      # Package setup file
└── pyproject.toml                # Project configuration
```

## Key Concepts

### Video Capture and Processing

The `VideoCapture` class in `src/vision/capture.py` handles webcam access and frame processing. It runs in a separate thread to avoid blocking the UI, and provides methods for adding frame processors and retrieving processed frames.

### Posture Detection

The `PostureDetector` class in `src/vision/pose_detection.py` detects sitting and standing postures using MediaPipe's pose estimation. It analyzes the relative positions of body landmarks to determine the user's posture.

### Drinking Detection

The `DrinkingDetector` class in `src/vision/action_detection.py` detects drinking actions using MediaPipe's hand and face tracking. It analyzes the proximity of the hand to the face to determine if the user is drinking.

### Data Storage

The `Database` class in `src/data/database.py` handles data storage using SQLite. It provides methods for storing and retrieving events, as well as generating daily summaries.

### Data Analysis

The `HealthAnalytics` class in `src/data/analytics.py` analyzes health data and generates statistics. It provides methods for calculating standing and drinking statistics, as well as generating health scores and recommendations.

### User Interface

The UI is built using PyQt6 and follows a tab-based design. The `MainWindow` class in `src/ui/main_window.py` defines the main application window, which contains tabs for the dashboard and settings.

### Configuration

The `Config` class in `src/utils/config.py` manages application configuration. It provides methods for getting and setting configuration values, as well as loading and saving the configuration to a file.

### Notifications

The `NotificationManager` class in `src/utils/notifications.py` handles system notifications. It provides methods for sending notifications and checking if reminders should be sent.


### Detection Tuning

The accuracy of posture (standing/sitting) and drinking detection can be fine-tuned to specific user environments and behaviors by adjusting parameters in the configuration file.

**Configuration File:**

The primary configuration file is `config.json`, typically located in `~/.employee_health_monitor/config.json` (it's created with default values on first run if it doesn't exist). Changes to this file require an application restart to take effect.

**Key Parameters for Tuning:**

**1. Drinking Detection:**
   - Located under the `detection.drinking` section in `config.json`.
   - `drinking_confidence_threshold` (float, default: 0.55): The minimum score (0-1) required from the detection algorithm to consider a frame as part of a potential drinking action. Lowering this increases sensitivity (more detections, potentially more false positives). Raising it decreases sensitivity.
   - `min_drinking_frames` (integer, default: 5): The number of consecutive frames the `drinking_confidence_threshold` must be met to confirm a drinking event. Higher values make the detection more robust against brief, non-drinking hand-to-face movements but might miss very quick sips.
   - `history_size` (integer, default: 20): The number of recent frames over which detection scores are smoothed.
   - `hand_to_face_threshold` (float, default: 0.15): An internal threshold related to raw hand-to-face proximity, generally less impactful for tuning than `drinking_confidence_threshold`.

**2. Posture Detection (Standing/Sitting):**
   - Located under the `detection.posture` section in `config.json`.
   - `standing_threshold` (float, default: 0.65): The minimum score (0-1) from the posture algorithm for the user to be classified as "standing". Higher values require a more definitive standing posture.
   - `sitting_threshold` (float, default: 0.35): The maximum score (0-1) from the posture algorithm for the user to be classified as "sitting". Lower values require a more definitive sitting posture.
   - *Note on Transitioning*: The system uses a hysteresis mechanism. The user enters a "transitioning" state if their posture score falls between `sitting_threshold` and `standing_threshold` or briefly crosses one of these thresholds from an established state.
   - `history_size` (integer, default: 20): The number of recent frames over which posture scores are smoothed.

**General Tuning Advice:**
   - Adjust parameters one at a time and by small increments.
   - Test with diverse scenarios (different ways of drinking, standing up, various environmental conditions).
   - Use application logs (set `app.log_level` to `INFO` or `DEBUG` in `config.json`) to observe detection scores and state changes, which can help diagnose false positives/negatives.
   - The goal is to achieve a balance that correctly identifies most true events while minimizing false detections.


## Design Patterns

### Observer Pattern

The application uses the observer pattern for event handling. For example, the `MonitoringThread` class emits signals when events occur, and the UI components observe these signals to update the display.

### Singleton Pattern

The `Config` and `NotificationManager` classes use the singleton pattern to ensure that only one instance of each class exists.

### Factory Pattern

The `Database` class uses the factory pattern to create database connections.

### Strategy Pattern

The `VideoCapture` class uses the strategy pattern to allow different frame processors to be used.

## Testing

The application uses pytest for testing. Tests are organized by module:

- `tests/test_vision.py`: Tests for the vision module
- `tests/test_data.py`: Tests for the data module
- `tests/test_ui.py`: Tests for the UI module

## Coding Standards

The project follows the following coding standards:

- **PEP 8**: Python style guide
- **Type Hints**: All functions and methods should include type hints
- **Docstrings**: All modules, classes, and functions should have docstrings
- **Unit Tests**: All code should have unit tests

## Continuous Integration

The project uses GitHub Actions for continuous integration. The CI pipeline runs the following checks:

- **Linting**: Checks code style using flake8
- **Type Checking**: Checks type hints using mypy
- **Testing**: Runs unit tests using pytest

## Release Process

1. Update version number in `src/__init__.py` and `setup.py`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Build and upload the package to PyPI

## Contributing

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Troubleshooting

### Common Development Issues

#### ImportError: No module named 'src'

This error occurs when the Python interpreter cannot find the `src` module. Make sure you are running the application from the project root directory.

#### Camera Access Issues

If you're having issues with camera access, check that your camera is working and that you have the necessary permissions.

#### UI Rendering Issues

If you're having issues with UI rendering, check that you have the correct version of PyQt6 installed.

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Plotly Documentation](https://plotly.com/python/)
