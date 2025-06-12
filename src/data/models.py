"""
Data Models Module for Employee Health Monitoring System.

This module defines data structures and models used throughout the application.
"""
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

class PostureState(Enum):
    """Enum representing different posture states."""
    UNKNOWN = 0
    SITTING = 1
    STANDING = 2
    TRANSITIONING = 3


class DrinkingState(Enum):
    """Enum representing different drinking states."""
    NOT_DRINKING = 0
    POTENTIAL_DRINKING = 1
    DRINKING = 2


@dataclass
class HeartRateEvent:
    """Class for storing heart rate event data."""
    timestamp: float
    bpm: float
    confidence: float = 1.0
    
    @property
    def datetime(self) -> str:
        """Get formatted datetime string."""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "timestamp": self.timestamp,
            "bpm": self.bpm,
            "confidence": self.confidence,
            "datetime": self.datetime
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeartRateEvent':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            bpm=data["bpm"],
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class PostureEvent:
    """Class for storing posture event data."""
    state: PostureState
    timestamp: float
    confidence: float = 1.0
    duration: Optional[float] = None
    
    @property
    def datetime(self) -> str:
        """Get formatted datetime string."""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "state": self.state.name,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "duration": self.duration,
            "datetime": self.datetime
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostureEvent':
        """Create from dictionary."""
        return cls(
            state=PostureState[data["state"]] if isinstance(data["state"], str) else PostureState(data["state"]),
            timestamp=data["timestamp"],
            confidence=data.get("confidence", 1.0),
            duration=data.get("duration")
        )


@dataclass
class DrinkingEvent:
    """Class for storing drinking event data."""
    timestamp: float
    confidence: float = 1.0
    duration: float = 0.0
    
    @property
    def datetime(self) -> str:
        """Get formatted datetime string."""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "duration": self.duration,
            "datetime": self.datetime
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DrinkingEvent':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            confidence=data.get("confidence", 1.0),
            duration=data.get("duration", 0.0)
        )


@dataclass
class DailySummary:
    """Class for storing daily health summary data."""
    date: str  # YYYY-MM-DD format
    total_standing_time: float = 0.0
    standing_count: int = 0
    drinking_count: int = 0
    avg_heart_rate: float = 0.0
    min_heart_rate: float = 0.0
    max_heart_rate: float = 0.0
    heart_rate_count: int = 0
    summary_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def date_obj(self) -> datetime.date:
        """Get date object."""
        return datetime.strptime(self.date, '%Y-%m-%d').date()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        result = {
            "date": self.date,
            "total_standing_time": self.total_standing_time,
            "standing_count": self.standing_count,
            "drinking_count": self.drinking_count,
        }
        
        # Add any additional summary data
        result.update(self.summary_data)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DailySummary':
        """Create from dictionary."""
        # Extract known fields
        known_fields = {"date", "total_standing_time", "standing_count", "drinking_count"}
        base_data = {k: data[k] for k in known_fields if k in data}
        
        # Extract additional fields as summary_data
        summary_data = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(**base_data, summary_data=summary_data)


@dataclass
class WeeklySummary:
    """Class for storing weekly health summary data."""
    week_start_date: str  # YYYY-MM-DD format
    week_end_date: str  # YYYY-MM-DD format
    total_standing_time: float = 0.0
    total_standing_count: int = 0
    total_drinking_count: int = 0
    avg_standing_time_per_day: float = 0.0
    avg_standing_count_per_day: float = 0.0
    avg_drinking_count_per_day: float = 0.0
    daily_summaries: List[DailySummary] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "week_start_date": self.week_start_date,
            "week_end_date": self.week_end_date,
            "total_standing_time": self.total_standing_time,
            "total_standing_count": self.total_standing_count,
            "total_drinking_count": self.total_drinking_count,
            "avg_standing_time_per_day": self.avg_standing_time_per_day,
            "avg_standing_count_per_day": self.avg_standing_count_per_day,
            "avg_drinking_count_per_day": self.avg_drinking_count_per_day,
            "daily_summaries": [summary.to_dict() for summary in self.daily_summaries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeeklySummary':
        """Create from dictionary."""
        # Handle daily_summaries separately
        daily_summaries_data = data.pop("daily_summaries", [])
        daily_summaries = [
            DailySummary.from_dict(summary) for summary in daily_summaries_data
        ]
        
        # Create instance with remaining data
        instance = cls(**data)
        instance.daily_summaries = daily_summaries
        
        return instance


@dataclass
class HealthScore:
    """Class for storing health score data."""
    date: str  # YYYY-MM-DD format
    overall_score: float = 0.0
    standing_score: float = 0.0
    drinking_score: float = 0.0
    heart_rate_score: float = 0.0
    standing_time: float = 0.0
    standing_count: int = 0
    drinking_count: int = 0
    avg_heart_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "date": self.date,
            "overall_score": self.overall_score,
            "standing_score": self.standing_score,
            "drinking_score": self.drinking_score,
            "standing_time": self.standing_time,
            "standing_count": self.standing_count,
            "drinking_count": self.drinking_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthScore':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class UserSettings:
    """Class for storing user settings."""
    standing_goal_minutes: int = 120  # 2 hours per day
    drinking_goal_count: int = 8  # 8 glasses per day
    heart_rate_min: int = 60  # Minimum healthy heart rate
    heart_rate_max: int = 100  # Maximum healthy heart rate
    notification_enabled: bool = True
    standing_reminder_minutes: int = 60  # Remind after 60 minutes of sitting
    drinking_reminder_minutes: int = 90  # Remind every 90 minutes
    start_monitoring_on_launch: bool = False
    camera_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "standing_goal_minutes": self.standing_goal_minutes,
            "drinking_goal_count": self.drinking_goal_count,
            "notification_enabled": self.notification_enabled,
            "standing_reminder_minutes": self.standing_reminder_minutes,
            "drinking_reminder_minutes": self.drinking_reminder_minutes,
            "start_monitoring_on_launch": self.start_monitoring_on_launch,
            "camera_id": self.camera_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSettings':
        """Create from dictionary."""
        return cls(**data)


# Example usage
if __name__ == "__main__":
    # Create a posture event
    posture_event = PostureEvent(
        state=PostureState.STANDING,
        timestamp=time.time(),
        confidence=0.95,
        duration=300.0
    )
    
    # Convert to dictionary
    posture_dict = posture_event.to_dict()
    print(f"Posture event: {posture_dict}")
    
    # Create from dictionary
    posture_event2 = PostureEvent.from_dict(posture_dict)
    print(f"Recreated posture event: {posture_event2}")
    
    # Create a drinking event
    drinking_event = DrinkingEvent(
        timestamp=time.time(),
        confidence=0.9,
        duration=5.0
    )
    
    # Convert to dictionary
    drinking_dict = drinking_event.to_dict()
    print(f"Drinking event: {drinking_dict}")
    
    # Create user settings
    settings = UserSettings()
    print(f"Default settings: {settings.to_dict()}")
