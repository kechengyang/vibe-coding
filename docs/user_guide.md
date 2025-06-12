# Employee Health Monitoring System - User Guide

## Introduction

The Employee Health Monitoring System is a desktop application designed to help you maintain healthy work habits by monitoring your standing and drinking behaviors throughout the day. The application uses your computer's webcam to detect when you stand up and when you drink water, and provides statistics and reminders to help you maintain good health habits.

## Privacy

The Employee Health Monitoring System is designed with privacy in mind:

- All video processing happens locally on your computer
- No images or video are stored or transmitted
- No personal data is collected or shared
- You have full control over when monitoring is active

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam access
- Windows, macOS, or Linux operating system

### Setup

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

## Using the Application

### Main Window

The main window of the application consists of a toolbar at the top and a tab widget below.

#### Toolbar

The toolbar contains the following buttons:

- **Start Monitoring**: Start monitoring your standing and drinking behaviors
- **Stop Monitoring**: Stop monitoring
- **Exit**: Exit the application

#### Dashboard Tab

The Dashboard tab displays statistics and visualizations of your standing and drinking behaviors:

- **Standing Time**: Total time spent standing
- **Standing Sessions**: Number of times you stood up
- **Water Intake**: Number of times you drank water
- **Health Score**: Overall health score based on your behaviors

The dashboard also includes charts showing your standing time, water intake, and health score over time.

#### Settings Tab

The Settings tab allows you to configure the application:

- **General Settings**: General application settings
- **Health Goals**: Set your daily standing and drinking goals
- **Notifications**: Configure notification settings
- **Camera Settings**: Configure camera settings

### Starting Monitoring

To start monitoring your standing and drinking behaviors:

1. Click the **Start Monitoring** button in the toolbar
2. Grant camera permissions when prompted
3. The system will begin monitoring automatically

### Stopping Monitoring

To stop monitoring:

1. Click the **Stop Monitoring** button in the toolbar

### Viewing Statistics

The Dashboard tab displays statistics and visualizations of your standing and drinking behaviors. You can select different time periods to view your statistics:

- **Today**: Statistics for today
- **Yesterday**: Statistics for yesterday
- **Last 7 Days**: Statistics for the last 7 days
- **Last 30 Days**: Statistics for the last 30 days

### Configuring Settings

The Settings tab allows you to configure the application:

#### General Settings

- **Start monitoring on application launch**: Automatically start monitoring when the application starts

#### Health Goals

- **Daily standing goal**: Set your daily standing goal in minutes
- **Daily water intake goal**: Set your daily water intake goal in number of times

#### Notifications

- **Enable notifications**: Enable or disable notifications
- **Remind to stand after sitting for**: Set how long to wait before reminding you to stand up
- **Remind to drink water every**: Set how often to remind you to drink water
- **Test Notification**: Send a test notification

#### Camera Settings

- **Select camera**: Select which camera to use
- **Test Camera**: Test the selected camera

## Notifications

The application can send notifications to remind you to stand up and drink water:

- **Standing Reminder**: Reminds you to stand up after sitting for a specified time
- **Drinking Reminder**: Reminds you to drink water at specified intervals
- **Achievement Notifications**: Notifies you when you achieve your standing and drinking goals

## Health Score

The application calculates a health score based on your standing and drinking behaviors:

- **Standing Score**: Based on your standing time and frequency
- **Drinking Score**: Based on your water intake
- **Overall Score**: Weighted combination of standing and drinking scores

## Recommendations

The application provides recommendations based on your behaviors to help you improve your health habits.

## Troubleshooting

### Camera Access

If the application cannot access your camera:

1. Check that your camera is connected and working
2. Check that you have granted camera permissions to the application
3. Try selecting a different camera in the Settings tab

### Notifications

If you are not receiving notifications:

1. Check that notifications are enabled in the Settings tab
2. Check that notifications are enabled in your operating system settings

### Performance

If the application is running slowly:

1. Close other applications that may be using your camera
2. Reduce the resolution of the camera in the Settings tab
3. Ensure your computer meets the minimum system requirements

## Support

If you encounter any issues or have questions, please:

1. Check the [FAQ](faq.md) for common questions and answers
2. Check the [GitHub Issues](https://github.com/your-organization/employee-health-monitoring/issues) for known issues
3. Submit a new issue if your problem is not already reported

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
