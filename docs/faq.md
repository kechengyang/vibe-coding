# Employee Health Monitoring System - Frequently Asked Questions

## General Questions

### What is the Employee Health Monitoring System?

The Employee Health Monitoring System is a desktop application that uses computer vision to monitor and promote healthy work habits. It detects when you stand up from your desk and when you drink water, and provides statistics and reminders to help you maintain good health habits.

### How does it work?

The application uses your computer's webcam and computer vision algorithms to detect your posture (sitting or standing) and when you drink water. It records these events and provides statistics and visualizations to help you track your health habits.

### Is it free to use?

Yes, the Employee Health Monitoring System is open-source software released under the MIT License, which means it is free to use, modify, and distribute.

## Privacy and Security

### Does the application record or store video?

No, the application does not record or store any video. All video processing happens locally on your computer, and the raw video frames are discarded after processing.

### Does the application send data to the internet?

No, the application does not send any data to the internet. All data is stored locally on your computer.

### What data does the application store?

The application stores the following data:
- Timestamps of when you stand up and sit down
- Timestamps of when you drink water
- Daily summaries of your standing and drinking behaviors
- Your application settings

### Where is the data stored?

The data is stored in a SQLite database file in the `.employee_health_monitor` directory in your home directory.

### Can I delete my data?

Yes, you can delete your data by deleting the database file in the `.employee_health_monitor` directory in your home directory.

## Installation and Setup

### What are the system requirements?

- Python 3.8 or higher
- Webcam access
- Windows, macOS, or Linux operating system

### How do I install the application?

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

4. Start the application:
```bash
python -m src.main
```

### Can I use the application without a webcam?

No, the application requires a webcam to detect your posture and drinking behaviors.

### Can I use an external webcam?

Yes, you can use any webcam that is recognized by your operating system. You can select which camera to use in the Settings tab.

## Usage

### How do I start monitoring?

Click the "Start Monitoring" button in the toolbar.

### How do I stop monitoring?

Click the "Stop Monitoring" button in the toolbar.

### Can I minimize the application while monitoring?

Yes, the application will continue monitoring even when minimized.

### Does the application work when my computer is locked or in sleep mode?

No, the application cannot monitor when your computer is locked or in sleep mode.

### How accurate is the detection?

The detection accuracy depends on various factors such as lighting conditions, camera quality, and your position relative to the camera. In optimal conditions, the detection accuracy is typically above 90%.

## Features

### What statistics does the application provide?

The application provides the following statistics:
- Total standing time
- Number of standing sessions
- Number of times you drank water
- Overall health score

### How is the health score calculated?

The health score is calculated based on your standing time, standing frequency, and water intake. The score ranges from 0 to 100, with higher scores indicating better health habits.

### Can I customize the reminders?

Yes, you can customize the reminders in the Settings tab:
- Enable or disable notifications
- Set how long to wait before reminding you to stand up
- Set how often to remind you to drink water

### Can I export my data?

Currently, the application does not have a built-in export feature. However, since the data is stored in a SQLite database, you can use third-party tools to export the data.

## Troubleshooting

### The application cannot access my camera

1. Check that your camera is connected and working
2. Check that you have granted camera permissions to the application
3. Try selecting a different camera in the Settings tab

### I'm not receiving notifications

1. Check that notifications are enabled in the Settings tab
2. Check that notifications are enabled in your operating system settings

### The application is running slowly

1. Close other applications that may be using your camera
2. Reduce the resolution of the camera in the Settings tab
3. Ensure your computer meets the minimum system requirements

### The detection is not accurate

1. Ensure good lighting conditions
2. Position yourself properly in front of the camera
3. Adjust the detection thresholds in the configuration file

## Development

### How can I contribute to the project?

You can contribute to the project by:
1. Reporting bugs
2. Suggesting features
3. Submitting pull requests

### Where can I find the source code?

The source code is available on GitHub: [https://github.com/your-organization/employee-health-monitoring](https://github.com/your-organization/employee-health-monitoring)

### What technologies does the application use?

The application uses the following technologies:
- Python for the core logic
- OpenCV and MediaPipe for computer vision
- PyQt6 for the user interface
- SQLite for data storage
- Matplotlib and Plotly for data visualization

### How can I build a standalone executable?

You can build a standalone executable using PyInstaller:

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller --name="Employee Health Monitor" --windowed --onefile src/main.py
```

The executable will be created in the `dist` directory.
