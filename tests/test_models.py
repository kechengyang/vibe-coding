"""
Tests for the data models module.
"""
import time
import unittest
from datetime import datetime

from src.data.models import (
    PostureState, DrinkingState, PostureEvent, DrinkingEvent,
    DailySummary, WeeklySummary, HealthScore, UserSettings
)


class TestModels(unittest.TestCase):
    """Test cases for the data models."""

    def test_posture_event(self):
        """Test PostureEvent class."""
        # Create a posture event
        timestamp = time.time()
        event = PostureEvent(
            state=PostureState.STANDING,
            timestamp=timestamp,
            confidence=0.95,
            duration=300.0
        )
        
        # Test properties
        self.assertEqual(event.state, PostureState.STANDING)
        self.assertEqual(event.timestamp, timestamp)
        self.assertEqual(event.confidence, 0.95)
        self.assertEqual(event.duration, 300.0)
        self.assertEqual(event.datetime, datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
        
        # Test to_dict method
        event_dict = event.to_dict()
        self.assertEqual(event_dict["state"], "STANDING")
        self.assertEqual(event_dict["timestamp"], timestamp)
        self.assertEqual(event_dict["confidence"], 0.95)
        self.assertEqual(event_dict["duration"], 300.0)
        
        # Test from_dict method
        event2 = PostureEvent.from_dict(event_dict)
        self.assertEqual(event2.state, PostureState.STANDING)
        self.assertEqual(event2.timestamp, timestamp)
        self.assertEqual(event2.confidence, 0.95)
        self.assertEqual(event2.duration, 300.0)

    def test_drinking_event(self):
        """Test DrinkingEvent class."""
        # Create a drinking event
        timestamp = time.time()
        event = DrinkingEvent(
            timestamp=timestamp,
            confidence=0.9,
            duration=5.0
        )
        
        # Test properties
        self.assertEqual(event.timestamp, timestamp)
        self.assertEqual(event.confidence, 0.9)
        self.assertEqual(event.duration, 5.0)
        self.assertEqual(event.datetime, datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
        
        # Test to_dict method
        event_dict = event.to_dict()
        self.assertEqual(event_dict["timestamp"], timestamp)
        self.assertEqual(event_dict["confidence"], 0.9)
        self.assertEqual(event_dict["duration"], 5.0)
        
        # Test from_dict method
        event2 = DrinkingEvent.from_dict(event_dict)
        self.assertEqual(event2.timestamp, timestamp)
        self.assertEqual(event2.confidence, 0.9)
        self.assertEqual(event2.duration, 5.0)

    def test_daily_summary(self):
        """Test DailySummary class."""
        # Create a daily summary
        summary = DailySummary(
            date="2025-06-12",
            total_standing_time=3600.0,
            standing_count=5,
            drinking_count=8,
            summary_data={"extra_field": "value"}
        )
        
        # Test properties
        self.assertEqual(summary.date, "2025-06-12")
        self.assertEqual(summary.total_standing_time, 3600.0)
        self.assertEqual(summary.standing_count, 5)
        self.assertEqual(summary.drinking_count, 8)
        self.assertEqual(summary.summary_data, {"extra_field": "value"})
        
        # Test to_dict method
        summary_dict = summary.to_dict()
        self.assertEqual(summary_dict["date"], "2025-06-12")
        self.assertEqual(summary_dict["total_standing_time"], 3600.0)
        self.assertEqual(summary_dict["standing_count"], 5)
        self.assertEqual(summary_dict["drinking_count"], 8)
        self.assertEqual(summary_dict["extra_field"], "value")
        
        # Test from_dict method
        summary2 = DailySummary.from_dict(summary_dict)
        self.assertEqual(summary2.date, "2025-06-12")
        self.assertEqual(summary2.total_standing_time, 3600.0)
        self.assertEqual(summary2.standing_count, 5)
        self.assertEqual(summary2.drinking_count, 8)
        self.assertEqual(summary2.summary_data, {"extra_field": "value"})

    def test_user_settings(self):
        """Test UserSettings class."""
        # Create user settings with default values
        settings = UserSettings()
        
        # Test default values
        self.assertEqual(settings.standing_goal_minutes, 120)
        self.assertEqual(settings.drinking_goal_count, 8)
        self.assertTrue(settings.notification_enabled)
        self.assertEqual(settings.standing_reminder_minutes, 60)
        self.assertEqual(settings.drinking_reminder_minutes, 90)
        self.assertFalse(settings.start_monitoring_on_launch)
        self.assertEqual(settings.camera_id, 0)
        
        # Create user settings with custom values
        custom_settings = UserSettings(
            standing_goal_minutes=180,
            drinking_goal_count=10,
            notification_enabled=False,
            standing_reminder_minutes=45,
            drinking_reminder_minutes=60,
            start_monitoring_on_launch=True,
            camera_id=1
        )
        
        # Test custom values
        self.assertEqual(custom_settings.standing_goal_minutes, 180)
        self.assertEqual(custom_settings.drinking_goal_count, 10)
        self.assertFalse(custom_settings.notification_enabled)
        self.assertEqual(custom_settings.standing_reminder_minutes, 45)
        self.assertEqual(custom_settings.drinking_reminder_minutes, 60)
        self.assertTrue(custom_settings.start_monitoring_on_launch)
        self.assertEqual(custom_settings.camera_id, 1)
        
        # Test to_dict method
        settings_dict = custom_settings.to_dict()
        self.assertEqual(settings_dict["standing_goal_minutes"], 180)
        self.assertEqual(settings_dict["drinking_goal_count"], 10)
        self.assertFalse(settings_dict["notification_enabled"])
        self.assertEqual(settings_dict["standing_reminder_minutes"], 45)
        self.assertEqual(settings_dict["drinking_reminder_minutes"], 60)
        self.assertTrue(settings_dict["start_monitoring_on_launch"])
        self.assertEqual(settings_dict["camera_id"], 1)
        
        # Test from_dict method
        settings2 = UserSettings.from_dict(settings_dict)
        self.assertEqual(settings2.standing_goal_minutes, 180)
        self.assertEqual(settings2.drinking_goal_count, 10)
        self.assertFalse(settings2.notification_enabled)
        self.assertEqual(settings2.standing_reminder_minutes, 45)
        self.assertEqual(settings2.drinking_reminder_minutes, 60)
        self.assertTrue(settings2.start_monitoring_on_launch)
        self.assertEqual(settings2.camera_id, 1)


if __name__ == '__main__':
    unittest.main()
