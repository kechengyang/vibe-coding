"""
Notifications Module for Employee Health Monitoring System.

This module handles system notifications for health reminders.
"""
import os
import time
import logging
import platform
import subprocess
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .config import get_config

logger = logging.getLogger("employee_health_monitor.utils.notifications")

class NotificationManager:
    """Class for managing system notifications."""
    
    def __init__(self):
        """Initialize the notification manager."""
        self.config = get_config()
        self.last_standing_reminder = 0
        self.last_drinking_reminder = 0
        self.sitting_start_time = 0
        self.system = platform.system()
        
        logger.info(f"NotificationManager initialized on {self.system}")
    
    def send_notification(
        self, 
        title: str, 
        message: str, 
        urgency: str = "normal",
        icon: Optional[str] = None
    ) -> bool:
        """Send a system notification.
        
        Args:
            title: Notification title
            message: Notification message
            urgency: Notification urgency ('low', 'normal', 'critical')
            icon: Path to icon image (optional)
            
        Returns:
            bool: True if notification was sent successfully
        """
        # Check if notifications are enabled
        if not self.config.get("user_settings.notification_enabled", True):
            logger.debug("Notifications are disabled")
            return False
        
        try:
            # Send notification based on platform
            if self.system == "Linux":
                return self._send_linux_notification(title, message, urgency, icon)
            elif self.system == "Darwin":  # macOS
                return self._send_macos_notification(title, message, icon)
            elif self.system == "Windows":
                return self._send_windows_notification(title, message, icon)
            else:
                logger.warning(f"Unsupported platform: {self.system}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False
    
    def _send_linux_notification(
        self, 
        title: str, 
        message: str, 
        urgency: str = "normal",
        icon: Optional[str] = None
    ) -> bool:
        """Send a notification on Linux using notify-send.
        
        Args:
            title: Notification title
            message: Notification message
            urgency: Notification urgency ('low', 'normal', 'critical')
            icon: Path to icon image (optional)
            
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            cmd = ["notify-send", title, message, f"--urgency={urgency}"]
            
            if icon and os.path.exists(icon):
                cmd.extend(["--icon", icon])
            
            subprocess.run(cmd, check=True)
            logger.debug(f"Linux notification sent: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Linux notification: {str(e)}")
            return False
    
    def _send_macos_notification(
        self, 
        title: str, 
        message: str, 
        icon: Optional[str] = None
    ) -> bool:
        """Send a notification on macOS using osascript.
        
        Args:
            title: Notification title
            message: Notification message
            icon: Path to icon image (not used on macOS)
            
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            # Escape double quotes in title and message
            title = title.replace('"', '\\"')
            message = message.replace('"', '\\"')
            
            script = f'''
            display notification "{message}" with title "{title}"
            '''
            
            subprocess.run(["osascript", "-e", script], check=True)
            logger.debug(f"macOS notification sent: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending macOS notification: {str(e)}")
            return False
    
    def _send_windows_notification(
        self, 
        title: str, 
        message: str, 
        icon: Optional[str] = None
    ) -> bool:
        """Send a notification on Windows using PowerShell.
        
        Args:
            title: Notification title
            message: Notification message
            icon: Path to icon image (not used on Windows)
            
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            # Escape single quotes in title and message
            title = title.replace("'", "''")
            message = message.replace("'", "''")
            
            # PowerShell script to show notification
            script = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

            $app_id = 'EmployeeHealthMonitor'
            $template = @"
            <toast>
                <visual>
                    <binding template="ToastText02">
                        <text id="1">{title}</text>
                        <text id="2">{message}</text>
                    </binding>
                </visual>
            </toast>
            "@

            $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
            $xml.LoadXml($template)
            $toast = New-Object Windows.UI.Notifications.ToastNotification $xml
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app_id).Show($toast)
            '''
            
            # Run PowerShell script
            subprocess.run(["powershell", "-Command", script], check=True)
            logger.debug(f"Windows notification sent: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Windows notification: {str(e)}")
            return False
    
    def check_and_send_reminders(self, posture_state: str, sitting_duration: float) -> None:
        """Check if reminders should be sent and send them if needed.
        
        Args:
            posture_state: Current posture state ('SITTING', 'STANDING', etc.)
            sitting_duration: Duration of current sitting session in seconds
        """
        now = time.time()
        
        # Update sitting start time
        if posture_state == "SITTING":
            if self.sitting_start_time == 0:
                self.sitting_start_time = now
        else:
            self.sitting_start_time = 0
        
        # Check for standing reminder
        if posture_state == "SITTING" and self.sitting_start_time > 0:
            standing_reminder_minutes = self.config.get("user_settings.standing_reminder_minutes", 60)
            standing_reminder_seconds = standing_reminder_minutes * 60
            
            # Send reminder if sitting for too long and enough time has passed since last reminder
            if (sitting_duration >= standing_reminder_seconds and 
                now - self.last_standing_reminder >= standing_reminder_seconds):
                
                # Send standing reminder
                self.send_notification(
                    "Time to Stand Up",
                    f"You've been sitting for {int(sitting_duration / 60)} minutes. Take a short standing break!",
                    "normal"
                )
                
                self.last_standing_reminder = now
                logger.info(f"Standing reminder sent after {int(sitting_duration / 60)} minutes of sitting")
        
        # Check for drinking reminder
        drinking_reminder_minutes = self.config.get("user_settings.drinking_reminder_minutes", 90)
        drinking_reminder_seconds = drinking_reminder_minutes * 60
        
        if now - self.last_drinking_reminder >= drinking_reminder_seconds:
            # Send drinking reminder
            self.send_notification(
                "Hydration Reminder",
                "Remember to drink water regularly throughout the day!",
                "normal"
            )
            
            self.last_drinking_reminder = now
            logger.info("Drinking reminder sent")
    
    def send_standing_achievement(self, duration_minutes: float) -> None:
        """Send a notification for standing achievement.
        
        Args:
            duration_minutes: Standing duration in minutes
        """
        self.send_notification(
            "Standing Goal Progress",
            f"Great job! You stood for {int(duration_minutes)} minutes.",
            "normal"
        )
        
        logger.info(f"Standing achievement notification sent: {int(duration_minutes)} minutes")
    
    def send_drinking_achievement(self, count: int) -> None:
        """Send a notification for drinking achievement.
        
        Args:
            count: Number of drinking events today
        """
        self.send_notification(
            "Hydration Goal Progress",
            f"You've had water {count} times today. Keep it up!",
            "normal"
        )
        
        logger.info(f"Drinking achievement notification sent: {count} times")
    
    def send_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Send a notification with daily summary.
        
        Args:
            summary: Daily summary data
        """
        standing_time = summary.get("total_standing_time", 0) / 60  # Convert to minutes
        standing_count = summary.get("standing_count", 0)
        drinking_count = summary.get("drinking_count", 0)
        
        message = (
            f"Today's summary:\n"
            f"• Standing: {int(standing_time)} minutes ({standing_count} sessions)\n"
            f"• Hydration: {drinking_count} times"
        )
        
        self.send_notification(
            "Daily Health Summary",
            message,
            "normal"
        )
        
        logger.info("Daily summary notification sent")


# Global notification manager instance
_notification_manager = None

def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance.
    
    Returns:
        NotificationManager: Global notification manager instance
    """
    global _notification_manager
    
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    
    return _notification_manager


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get notification manager
    notification_manager = get_notification_manager()
    
    # Send a test notification
    notification_manager.send_notification(
        "Test Notification",
        "This is a test notification from the Employee Health Monitoring System.",
        "normal"
    )
    
    # Simulate sitting for a long time
    notification_manager.check_and_send_reminders("SITTING", 3600)  # 1 hour
    
    # Simulate standing achievement
    notification_manager.send_standing_achievement(30)  # 30 minutes
    
    # Simulate drinking achievement
    notification_manager.send_drinking_achievement(5)  # 5 times
