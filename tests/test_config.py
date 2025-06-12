"""
Tests for the configuration module.
"""
import os
import tempfile
import unittest
import json

from src.utils.config import Config
from src.data.models import UserSettings


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.json")
        self.config = Config(self.config_path)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_default_config(self):
        """Test default configuration values."""
        # Check some default values
        self.assertEqual(self.config.get("app.name"), "Employee Health Monitor")
        self.assertEqual(self.config.get("app.version"), "0.1.0")
        self.assertEqual(self.config.get("camera.id"), 0)
        self.assertEqual(self.config.get("camera.resolution"), [640, 480])
        self.assertEqual(self.config.get("detection.posture.standing_threshold"), 0.7)
        self.assertEqual(self.config.get("detection.drinking.drinking_confidence_threshold"), 0.7)
        self.assertEqual(self.config.get("ui.theme"), "light")

    def test_set_and_get(self):
        """Test setting and getting configuration values."""
        # Set a value
        self.config.set("ui.theme", "dark")
        
        # Get the value
        theme = self.config.get("ui.theme")
        
        # Verify value
        self.assertEqual(theme, "dark")
        
        # Set a nested value that doesn't exist yet
        self.config.set("new.nested.value", 42)
        
        # Get the value
        value = self.config.get("new.nested.value")
        
        # Verify value
        self.assertEqual(value, 42)
        
        # Get a non-existent value with default
        value = self.config.get("non.existent.key", "default")
        
        # Verify default value is returned
        self.assertEqual(value, "default")

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        # Set some values
        self.config.set("ui.theme", "dark")
        self.config.set("camera.id", 1)
        
        # Set user settings
        user_settings = UserSettings(
            standing_goal_minutes=180,
            drinking_goal_count=10,
            notification_enabled=False
        )
        self.config.set_user_settings(user_settings)
        
        # Save configuration
        self.assertTrue(self.config.save())
        
        # Verify file exists
        self.assertTrue(os.path.exists(self.config_path))
        
        # Create a new config instance to load the saved file
        new_config = Config(self.config_path)
        
        # Verify values were loaded
        self.assertEqual(new_config.get("ui.theme"), "dark")
        self.assertEqual(new_config.get("camera.id"), 1)
        
        # Verify user settings were loaded
        loaded_settings = new_config.get_user_settings()
        self.assertEqual(loaded_settings.standing_goal_minutes, 180)
        self.assertEqual(loaded_settings.drinking_goal_count, 10)
        self.assertFalse(loaded_settings.notification_enabled)

    def test_user_settings(self):
        """Test user settings handling."""
        # Get default user settings
        settings = self.config.get_user_settings()
        
        # Verify default values
        self.assertEqual(settings.standing_goal_minutes, 120)
        self.assertEqual(settings.drinking_goal_count, 8)
        self.assertTrue(settings.notification_enabled)
        
        # Modify settings
        settings.standing_goal_minutes = 150
        settings.drinking_goal_count = 12
        
        # Update config with modified settings
        self.config.set_user_settings(settings)
        
        # Get settings again
        updated_settings = self.config.get_user_settings()
        
        # Verify updated values
        self.assertEqual(updated_settings.standing_goal_minutes, 150)
        self.assertEqual(updated_settings.drinking_goal_count, 12)

    def test_get_data_dir(self):
        """Test getting data directory path."""
        # Set a custom data directory
        test_data_dir = os.path.join(self.temp_dir.name, "test_data")
        self.config.set("app.data_dir", test_data_dir)
        
        # Get data directory
        data_dir = self.config.get_data_dir()
        
        # Verify data directory
        self.assertEqual(data_dir, test_data_dir)
        
        # Verify directory was created
        self.assertTrue(os.path.exists(test_data_dir))

    def test_get_db_path(self):
        """Test getting database file path."""
        # Set a custom data directory
        test_data_dir = os.path.join(self.temp_dir.name, "test_data")
        self.config.set("app.data_dir", test_data_dir)
        
        # Set a custom database filename
        test_db_filename = "test_db.sqlite"
        self.config.set("app.db_filename", test_db_filename)
        
        # Get database path
        db_path = self.config.get_db_path()
        
        # Verify database path
        expected_path = os.path.join(test_data_dir, test_db_filename)
        self.assertEqual(db_path, expected_path)


if __name__ == '__main__':
    unittest.main()
