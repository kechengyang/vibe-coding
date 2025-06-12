"""
Tests for the database module.
"""
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta

from src.data.database import Database


class TestDatabase(unittest.TestCase):
    """Test cases for the Database class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary database file
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self.db = Database(self.temp_db_path)

    def tearDown(self):
        """Clean up test environment."""
        self.db.close()
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)

    def test_add_standing_event(self):
        """Test adding a standing event."""
        # Add a standing event
        now = datetime.now().timestamp()
        event_id = self.db.add_standing_event(
            start_time=now - 300,  # 5 minutes ago
            end_time=now,
            duration=300,
            confidence=0.9
        )

        # Verify event was added
        self.assertIsNotNone(event_id)
        self.assertGreater(event_id, 0)

        # Get events
        events = self.db.get_standing_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["id"], event_id)
        self.assertEqual(events[0]["duration"], 300)
        self.assertEqual(events[0]["confidence"], 0.9)

    def test_add_drinking_event(self):
        """Test adding a drinking event."""
        # Add a drinking event
        now = datetime.now().timestamp()
        event_id = self.db.add_drinking_event(
            timestamp=now,
            duration=5,
            confidence=0.8
        )

        # Verify event was added
        self.assertIsNotNone(event_id)
        self.assertGreater(event_id, 0)

        # Get events
        events = self.db.get_drinking_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["id"], event_id)
        self.assertEqual(events[0]["duration"], 5)
        self.assertEqual(events[0]["confidence"], 0.8)

    def test_get_daily_summaries(self):
        """Test getting daily summaries."""
        # Add events for today
        now = datetime.now().timestamp()
        today = date.today()

        # Add standing event
        self.db.add_standing_event(
            start_time=now - 300,
            end_time=now,
            duration=300,
            confidence=0.9
        )

        # Add drinking event
        self.db.add_drinking_event(
            timestamp=now,
            duration=5,
            confidence=0.8
        )

        # Get daily summaries
        summaries = self.db.get_daily_summaries(today, today)
        
        # Verify summary
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["date"], today.strftime('%Y-%m-%d'))
        self.assertEqual(summaries[0]["total_standing_time"], 300)
        self.assertEqual(summaries[0]["standing_count"], 1)
        self.assertEqual(summaries[0]["drinking_count"], 1)

    def test_save_and_get_setting(self):
        """Test saving and retrieving settings."""
        # Save a setting
        self.db.save_setting("test_key", "test_value")
        
        # Get the setting
        value = self.db.get_setting("test_key")
        
        # Verify setting
        self.assertEqual(value, "test_value")
        
        # Get a non-existent setting
        value = self.db.get_setting("non_existent_key", "default_value")
        
        # Verify default value is returned
        self.assertEqual(value, "default_value")


if __name__ == '__main__':
    unittest.main()
