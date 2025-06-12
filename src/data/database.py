"""
Database Module for Employee Health Monitoring System.

This module handles data storage and retrieval using SQLite.
"""
import os
import sqlite3
import logging
import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger("employee_health_monitor.data.database")

class Database:
    """Class for handling database operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database.
        
        Args:
            db_path: Path to the database file. If None, uses default path.
        """
        if db_path is None:
            # Use default path in user's home directory
            db_dir = Path.home() / ".employee_health_monitor"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "health_data.db")
        
        self.db_path = db_path
        self.conn = None
        
        logger.info(f"Database initialized with path: {db_path}")
        
        # Initialize database
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            
            # Standing events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS standing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL NOT NULL,
                end_time REAL,
                duration REAL,
                confidence REAL
            )
            ''')
            
            # Drinking events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS drinking_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                duration REAL,
                confidence REAL
            )
            ''')
            
            # Heart rate events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS heart_rate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                bpm REAL NOT NULL,
                confidence REAL
            )
            ''')
            
            # Daily summary table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summaries (
                date TEXT PRIMARY KEY,
                total_standing_time REAL DEFAULT 0,
                standing_count INTEGER DEFAULT 0,
                drinking_count INTEGER DEFAULT 0,
                summary_data TEXT
            )
            ''')
            
            # Settings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            ''')
            
            self.conn.commit()
            logger.info("Database schema initialized")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            if self.conn:
                self.conn.close()
                self.conn = None
            raise
    
    def _ensure_connection(self) -> None:
        """Ensure database connection is established."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def add_standing_event(
        self, 
        start_time: float, 
        end_time: Optional[float] = None, 
        duration: Optional[float] = None, 
        confidence: float = 1.0
    ) -> int:
        """Add a standing event to the database.
        
        Args:
            start_time: Start time of the standing event (timestamp)
            end_time: End time of the standing event (timestamp)
            duration: Duration of the standing event in seconds
            confidence: Confidence level of the detection (0-1)
            
        Returns:
            int: ID of the inserted event
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            # Calculate duration if not provided
            if duration is None and end_time is not None:
                duration = end_time - start_time
            
            cursor.execute(
                '''
                INSERT INTO standing_events (start_time, end_time, duration, confidence)
                VALUES (?, ?, ?, ?)
                ''',
                (start_time, end_time, duration, confidence)
            )
            
            self.conn.commit()
            event_id = cursor.lastrowid
            
            logger.info(f"Added standing event with ID {event_id}, duration: {duration:.2f}s")
            
            # Update daily summary
            self._update_daily_summary_for_standing(start_time, duration)
            
            return event_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding standing event: {str(e)}")
            self.conn.rollback()
            raise
    
    def update_standing_event(self, event_id: int, end_time: float, duration: float) -> bool:
        """Update a standing event with end time and duration.
        
        Args:
            event_id: ID of the event to update
            end_time: End time of the standing event (timestamp)
            duration: Duration of the standing event in seconds
            
        Returns:
            bool: True if update was successful
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            # Get the existing event to update daily summary
            cursor.execute(
                "SELECT start_time FROM standing_events WHERE id = ?",
                (event_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Standing event with ID {event_id} not found")
                return False
            
            start_time = result[0]
            
            cursor.execute(
                '''
                UPDATE standing_events
                SET end_time = ?, duration = ?
                WHERE id = ?
                ''',
                (end_time, duration, event_id)
            )
            
            self.conn.commit()
            
            logger.info(f"Updated standing event with ID {event_id}, duration: {duration:.2f}s")
            
            # Update daily summary
            self._update_daily_summary_for_standing(start_time, duration)
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error updating standing event: {str(e)}")
            self.conn.rollback()
            return False
    
    def add_heart_rate_event(
        self,
        timestamp: float,
        bpm: float,
        confidence: float = 1.0
    ) -> int:
        """Add a heart rate event to the database.
        
        Args:
            timestamp: Time of the heart rate measurement (timestamp)
            bpm: Heart rate in beats per minute
            confidence: Confidence level of the detection (0-1)
            
        Returns:
            int: ID of the inserted event
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                '''
                INSERT INTO heart_rate_events (timestamp, bpm, confidence)
                VALUES (?, ?, ?)
                ''',
                (timestamp, bpm, confidence)
            )
            
            self.conn.commit()
            event_id = cursor.lastrowid
            
            logger.info(f"Added heart rate event with ID {event_id}, BPM: {bpm:.1f}")
            
            # Update daily summary
            self._update_daily_summary_for_heart_rate(timestamp, bpm)
            
            return event_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding heart rate event: {str(e)}")
            self.conn.rollback()
            raise
    
    def add_drinking_event(
        self, 
        timestamp: float, 
        duration: float = 0.0, 
        confidence: float = 1.0
    ) -> int:
        """Add a drinking event to the database.
        
        Args:
            timestamp: Time of the drinking event (timestamp)
            duration: Duration of the drinking event in seconds
            confidence: Confidence level of the detection (0-1)
            
        Returns:
            int: ID of the inserted event
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                '''
                INSERT INTO drinking_events (timestamp, duration, confidence)
                VALUES (?, ?, ?)
                ''',
                (timestamp, duration, confidence)
            )
            
            self.conn.commit()
            event_id = cursor.lastrowid
            
            logger.info(f"Added drinking event with ID {event_id}")
            
            # Update daily summary
            self._update_daily_summary_for_drinking(timestamp)
            
            return event_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding drinking event: {str(e)}")
            self.conn.rollback()
            raise
    
    def _update_daily_summary_for_standing(self, timestamp: float, duration: float) -> None:
        """Update daily summary for a standing event.
        
        Args:
            timestamp: Event timestamp
            duration: Event duration in seconds
        """
        if duration is None:
            return
        
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        
        try:
            cursor = self.conn.cursor()
            
            # Check if summary exists for this date
            cursor.execute(
                "SELECT total_standing_time, standing_count FROM daily_summaries WHERE date = ?",
                (date_str,)
            )
            result = cursor.fetchone()
            
            if result:
                # Update existing summary
                total_standing_time, standing_count = result
                total_standing_time += duration
                standing_count += 1
                
                cursor.execute(
                    '''
                    UPDATE daily_summaries
                    SET total_standing_time = ?, standing_count = ?
                    WHERE date = ?
                    ''',
                    (total_standing_time, standing_count, date_str)
                )
            else:
                # Create new summary
                cursor.execute(
                    '''
                    INSERT INTO daily_summaries (date, total_standing_time, standing_count)
                    VALUES (?, ?, ?)
                    ''',
                    (date_str, duration, 1)
                )
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Error updating daily summary for standing: {str(e)}")
            self.conn.rollback()
    
    def _update_daily_summary_for_drinking(self, timestamp: float) -> None:
        """Update daily summary for a drinking event.
        
        Args:
            timestamp: Event timestamp
        """
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        
        try:
            cursor = self.conn.cursor()
            
            # Check if summary exists for this date
            cursor.execute(
                "SELECT drinking_count FROM daily_summaries WHERE date = ?",
                (date_str,)
            )
            result = cursor.fetchone()
            
            if result:
                # Update existing summary
                drinking_count = result[0] + 1
                
                cursor.execute(
                    '''
                    UPDATE daily_summaries
                    SET drinking_count = ?
                    WHERE date = ?
                    ''',
                    (drinking_count, date_str)
                )
            else:
                # Create new summary
                cursor.execute(
                    '''
                    INSERT INTO daily_summaries (date, drinking_count)
                    VALUES (?, ?)
                    ''',
                    (date_str, 1)
                )
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Error updating daily summary for drinking: {str(e)}")
            self.conn.rollback()
    
    def _update_daily_summary_for_heart_rate(self, timestamp: float, bpm: float) -> None:
        """Update daily summary for a heart rate event.
        
        Args:
            timestamp: Event timestamp
            bpm: Heart rate in beats per minute
        """
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        
        try:
            cursor = self.conn.cursor()
            
            # Check if summary exists for this date
            cursor.execute(
                """
                SELECT avg_heart_rate, min_heart_rate, max_heart_rate, heart_rate_count, summary_data
                FROM daily_summaries WHERE date = ?
                """,
                (date_str,)
            )
            result = cursor.fetchone()
            
            if result:
                # Update existing summary
                avg_heart_rate, min_heart_rate, max_heart_rate, heart_rate_count, summary_data_json = result
                
                # Parse summary data
                summary_data = {}
                if summary_data_json:
                    try:
                        summary_data = json.loads(summary_data_json)
                    except json.JSONDecodeError:
                        pass
                
                # Calculate new values
                heart_rate_count += 1
                
                # Calculate new average
                if avg_heart_rate > 0:
                    avg_heart_rate = ((avg_heart_rate * (heart_rate_count - 1)) + bpm) / heart_rate_count
                else:
                    avg_heart_rate = bpm
                
                # Update min/max
                if min_heart_rate == 0 or bpm < min_heart_rate:
                    min_heart_rate = bpm
                
                if bpm > max_heart_rate:
                    max_heart_rate = bpm
                
                # Update summary data
                summary_data["avg_heart_rate"] = avg_heart_rate
                summary_data["min_heart_rate"] = min_heart_rate
                summary_data["max_heart_rate"] = max_heart_rate
                summary_data["heart_rate_count"] = heart_rate_count
                
                # Update database
                cursor.execute(
                    '''
                    UPDATE daily_summaries
                    SET summary_data = ?
                    WHERE date = ?
                    ''',
                    (json.dumps(summary_data), date_str)
                )
            else:
                # Create new summary with heart rate data
                summary_data = {
                    "avg_heart_rate": bpm,
                    "min_heart_rate": bpm,
                    "max_heart_rate": bpm,
                    "heart_rate_count": 1
                }
                
                cursor.execute(
                    '''
                    INSERT INTO daily_summaries (date, summary_data)
                    VALUES (?, ?)
                    ''',
                    (date_str, json.dumps(summary_data))
                )
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Error updating daily summary for heart rate: {str(e)}")
            self.conn.rollback()
    
    def get_standing_events(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get standing events within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of standing events as dictionaries
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT id, start_time, end_time, duration, confidence FROM standing_events"
            params = []
            
            # Add date filters if provided
            if start_date or end_date:
                conditions = []
                
                if start_date:
                    start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
                    conditions.append("start_time >= ?")
                    params.append(start_timestamp)
                
                if end_date:
                    # Add one day to include the entire end date
                    end_timestamp = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).timestamp()
                    conditions.append("start_time < ?")
                    params.append(end_timestamp)
                
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY start_time DESC"
            
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                event = {
                    "id": row[0],
                    "start_time": row[1],
                    "end_time": row[2],
                    "duration": row[3],
                    "confidence": row[4],
                    "start_datetime": datetime.fromtimestamp(row[1]).strftime('%Y-%m-%d %H:%M:%S'),
                }
                
                if row[2]:  # If end_time is not None
                    event["end_datetime"] = datetime.fromtimestamp(row[2]).strftime('%Y-%m-%d %H:%M:%S')
                
                events.append(event)
            
            return events
            
        except sqlite3.Error as e:
            logger.error(f"Error getting standing events: {str(e)}")
            return []
    
    def get_heart_rate_events(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get heart rate events within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of heart rate events as dictionaries
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT id, timestamp, bpm, confidence FROM heart_rate_events"
            params = []
            
            # Add date filters if provided
            if start_date or end_date:
                conditions = []
                
                if start_date:
                    start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
                    conditions.append("timestamp >= ?")
                    params.append(start_timestamp)
                
                if end_date:
                    # Add one day to include the entire end date
                    end_timestamp = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).timestamp()
                    conditions.append("timestamp < ?")
                    params.append(end_timestamp)
                
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "bpm": row[2],
                    "confidence": row[3],
                    "datetime": datetime.fromtimestamp(row[1]).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return events
            
        except sqlite3.Error as e:
            logger.error(f"Error getting heart rate events: {str(e)}")
            return []
    
    def get_drinking_events(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get drinking events within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of drinking events as dictionaries
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT id, timestamp, duration, confidence FROM drinking_events"
            params = []
            
            # Add date filters if provided
            if start_date or end_date:
                conditions = []
                
                if start_date:
                    start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
                    conditions.append("timestamp >= ?")
                    params.append(start_timestamp)
                
                if end_date:
                    # Add one day to include the entire end date
                    end_timestamp = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).timestamp()
                    conditions.append("timestamp < ?")
                    params.append(end_timestamp)
                
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "duration": row[2],
                    "confidence": row[3],
                    "datetime": datetime.fromtimestamp(row[1]).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return events
            
        except sqlite3.Error as e:
            logger.error(f"Error getting drinking events: {str(e)}")
            return []
    
    def get_daily_summaries(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get daily summaries within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of daily summaries as dictionaries
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            query = """
            SELECT date, total_standing_time, standing_count, drinking_count, summary_data
            FROM daily_summaries
            """
            params = []
            
            # Add date filters if provided
            if start_date or end_date:
                conditions = []
                
                if start_date:
                    conditions.append("date >= ?")
                    params.append(start_date.strftime('%Y-%m-%d'))
                
                if end_date:
                    conditions.append("date <= ?")
                    params.append(end_date.strftime('%Y-%m-%d'))
                
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY date DESC"
            
            cursor.execute(query, params)
            
            summaries = []
            for row in cursor.fetchall():
                summary = {
                    "date": row[0],
                    "total_standing_time": row[1],
                    "standing_count": row[2],
                    "drinking_count": row[3],
                }
                
                # Parse summary_data JSON if available
                if row[4]:
                    try:
                        summary_data = json.loads(row[4])
                        summary.update(summary_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in summary_data for date {row[0]}")
                
                summaries.append(summary)
            
            return summaries
            
        except sqlite3.Error as e:
            logger.error(f"Error getting daily summaries: {str(e)}")
            return []
    
    def get_weekly_summary(self, week_start_date: Optional[date] = None) -> Dict[str, Any]:
        """Get a weekly summary.
        
        Args:
            week_start_date: Start date of the week (defaults to most recent Monday)
            
        Returns:
            Weekly summary as a dictionary
        """
        # Determine week start and end dates
        if week_start_date is None:
            today = date.today()
            # Get the most recent Monday (0 = Monday in weekday())
            week_start_date = today - timedelta(days=today.weekday())
        
        week_end_date = week_start_date + timedelta(days=6)
        
        # Get daily summaries for the week
        daily_summaries = self.get_daily_summaries(week_start_date, week_end_date)
        
        # Aggregate data
        total_standing_time = sum(summary.get("total_standing_time", 0) for summary in daily_summaries)
        total_standing_count = sum(summary.get("standing_count", 0) for summary in daily_summaries)
        total_drinking_count = sum(summary.get("drinking_count", 0) for summary in daily_summaries)
        
        # Create weekly summary
        weekly_summary = {
            "week_start_date": week_start_date.strftime('%Y-%m-%d'),
            "week_end_date": week_end_date.strftime('%Y-%m-%d'),
            "total_standing_time": total_standing_time,
            "total_standing_count": total_standing_count,
            "total_drinking_count": total_drinking_count,
            "avg_standing_time_per_day": total_standing_time / 7 if total_standing_time > 0 else 0,
            "avg_standing_count_per_day": total_standing_count / 7 if total_standing_count > 0 else 0,
            "avg_drinking_count_per_day": total_drinking_count / 7 if total_drinking_count > 0 else 0,
            "daily_summaries": daily_summaries
        }
        
        return weekly_summary
    
    def save_setting(self, key: str, value: Any) -> bool:
        """Save a setting to the database.
        
        Args:
            key: Setting key
            value: Setting value (will be converted to JSON)
            
        Returns:
            bool: True if successful
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            # Convert value to JSON string
            value_json = json.dumps(value)
            
            cursor.execute(
                '''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES (?, ?)
                ''',
                (key, value_json)
            )
            
            self.conn.commit()
            logger.info(f"Saved setting: {key}")
            return True
            
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Error saving setting: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting from the database.
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value
        """
        self._ensure_connection()
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,)
            )
            
            result = cursor.fetchone()
            
            if result:
                # Parse JSON value
                return json.loads(result[0])
            else:
                return default
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Error getting setting: {str(e)}")
            return default
    
    def __enter__(self):
        """Context manager entry."""
        self._ensure_connection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create database
    with Database() as db:
        # Add some test data
        now = time.time()
        
        # Add standing event
        standing_id = db.add_standing_event(
            start_time=now - 300,  # 5 minutes ago
            end_time=now,
            duration=300,
            confidence=0.9
        )
        
        # Add drinking event
        drinking_id = db.add_drinking_event(
            timestamp=now - 600,  # 10 minutes ago
            duration=5,
            confidence=0.8
        )
        
        # Get events
        standing_events = db.get_standing_events()
        drinking_events = db.get_drinking_events()
        
        print(f"Standing events: {len(standing_events)}")
        print(f"Drinking events: {len(drinking_events)}")
        
        # Get daily summary
        today = date.today()
        daily_summaries = db.get_daily_summaries(today, today)
        
        print(f"Daily summary: {daily_summaries}")
