# Employee Health Monitoring System

An open-source desktop application that uses computer vision to monitor and promote healthy work habits.

## Overview

This application uses your computer's webcam to detect and track health-related behaviors while you work:
- Standing up duration tracking
- Water drinking detection

All processing happens locally on your device with a strong focus on privacy.

## Features

- **Privacy-First Design**: All video processing happens locally on your device. No images or video are stored or transmitted.
- **Standing Detection**: Automatically tracks when you stand up from your desk and for how long.
- **Drinking Detection**: Recognizes when you drink water and logs these events.
- **Health Analytics**: View trends and statistics about your healthy behaviors.
- **Customizable Reminders**: Optional notifications to encourage regular movement and hydration.

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam access
- Windows, macOS, or Linux operating system

### Setup

1. Clone this repository:
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

4. Start the application:
```bash
python -m src.main
```

## Usage

1. Launch the application
2. Grant camera permissions when prompted
3. The system will begin monitoring automatically
4. View your statistics in the dashboard
5. Configure settings as needed

## Privacy

- **Local Processing**: All video analysis happens on your local device
- **No Data Storage**: Raw video is never stored
- **Transparent Code**: Open-source so you can verify privacy claims
- **User Control**: Easy to pause monitoring or exit the application

## Development

### Setting Up Development Environment

1. Follow the installation steps above
2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- MediaPipe for pose estimation
- PyQt6 for the user interface
