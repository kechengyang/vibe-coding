"""
Analytics Module for Employee Health Monitoring System.

This module handles data analysis and statistics generation.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from .database import Database

logger = logging.getLogger("employee_health_monitor.data.analytics")

class HealthAnalytics:
    """Class for analyzing health monitoring data."""
    
    def __init__(self, database: Optional[Database] = None):
        """Initialize the analytics engine.
        
        Args:
            database: Database instance to use for data retrieval
        """
        self.db = database or Database()
        logger.info("HealthAnalytics initialized")
    
    def get_standing_stats(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get statistics for standing events.
        
        Args:
            start_date: Start date for analysis (inclusive)
            end_date: End date for analysis (inclusive)
            
        Returns:
            Dictionary of standing statistics
        """
        # Get standing events
        events = self.db.get_standing_events(start_date, end_date)
        
        if not events:
            return {
                "total_events": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "max_duration": 0,
                "min_duration": 0,
                "events_per_day": 0,
                "duration_per_day": 0,
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
            }
        
        # Extract durations
        durations = [event["duration"] for event in events if event["duration"] is not None]
        
        # Calculate date range
        if start_date and end_date:
            days = (end_date - start_date).days + 1
        elif events:
            # Use event dates if no explicit range provided
            timestamps = [event["start_time"] for event in events]
            min_date = date.fromtimestamp(min(timestamps))
            max_date = date.fromtimestamp(max(timestamps))
            days = (max_date - min_date).days + 1
        else:
            days = 1
        
        # Calculate statistics
        total_events = len(events)
        total_duration = sum(durations) if durations else 0
        avg_duration = np.mean(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        events_per_day = total_events / days
        duration_per_day = total_duration / days
        
        return {
            "total_events": total_events,
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "events_per_day": events_per_day,
            "duration_per_day": duration_per_day,
            "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
            "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
        }
    
    def get_heart_rate_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get statistics for heart rate events.
        
        Args:
            start_date: Start date for analysis (inclusive)
            end_date: End date for analysis (inclusive)
            
        Returns:
            Dictionary of heart rate statistics
        """
        # Get heart rate events
        events = self.db.get_heart_rate_events(start_date, end_date)
        
        if not events:
            return {
                "total_events": 0,
                "avg_bpm": 0,
                "min_bpm": 0,
                "max_bpm": 0,
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
            }
        
        # Extract BPM values
        bpm_values = [event["bpm"] for event in events]
        
        # Calculate statistics
        total_events = len(events)
        avg_bpm = np.mean(bpm_values) if bpm_values else 0
        min_bpm = min(bpm_values) if bpm_values else 0
        max_bpm = max(bpm_values) if bpm_values else 0
        
        # Calculate date range
        if start_date and end_date:
            days = (end_date - start_date).days + 1
        elif events:
            # Use event dates if no explicit range provided
            timestamps = [event["timestamp"] for event in events]
            min_date = date.fromtimestamp(min(timestamps))
            max_date = date.fromtimestamp(max(timestamps))
            days = (max_date - min_date).days + 1
        else:
            days = 1
        
        # Group events by hour to find patterns
        hour_counts = {}
        for event in events:
            hour = datetime.fromtimestamp(event["timestamp"]).hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        return {
            "total_events": total_events,
            "avg_bpm": avg_bpm,
            "min_bpm": min_bpm,
            "max_bpm": max_bpm,
            "events_per_day": total_events / days,
            "hour_distribution": hour_counts,
            "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
            "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
        }
    
    def get_drinking_stats(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get statistics for drinking events.
        
        Args:
            start_date: Start date for analysis (inclusive)
            end_date: End date for analysis (inclusive)
            
        Returns:
            Dictionary of drinking statistics
        """
        # Get drinking events
        events = self.db.get_drinking_events(start_date, end_date)
        
        if not events:
            return {
                "total_events": 0,
                "events_per_day": 0,
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
            }
        
        # Calculate date range
        if start_date and end_date:
            days = (end_date - start_date).days + 1
        elif events:
            # Use event dates if no explicit range provided
            timestamps = [event["timestamp"] for event in events]
            min_date = date.fromtimestamp(min(timestamps))
            max_date = date.fromtimestamp(max(timestamps))
            days = (max_date - min_date).days + 1
        else:
            days = 1
        
        # Calculate statistics
        total_events = len(events)
        events_per_day = total_events / days
        
        # Group events by hour to find patterns
        hour_counts = {}
        for event in events:
            hour = datetime.fromtimestamp(event["timestamp"]).hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours (hours with most drinking events)
        peak_hours = []
        if hour_counts:
            max_count = max(hour_counts.values())
            peak_hours = [hour for hour, count in hour_counts.items() if count == max_count]
        
        return {
            "total_events": total_events,
            "events_per_day": events_per_day,
            "hour_distribution": hour_counts,
            "peak_hours": peak_hours,
            "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
            "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
        }
    
    def get_daily_trend(
        self, 
        metric: str, 
        days: int = 30
    ) -> Dict[str, Any]:
        """Get daily trend data for a specific metric.
        
        Args:
            metric: Metric to analyze ('standing_time', 'standing_count', 'drinking_count')
            days: Number of days to include in the trend
            
        Returns:
            Dictionary with trend data
        """
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days-1)
        
        # Get daily summaries
        summaries = self.db.get_daily_summaries(start_date, end_date)
        
        # Create a complete date range
        date_range = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        
        # Initialize data with zeros
        data = {date_str: 0 for date_str in date_range}
        
        # Fill in actual data
        for summary in summaries:
            date_str = summary["date"]
            if date_str in data:
                if metric == 'standing_time':
                    data[date_str] = summary.get("total_standing_time", 0)
                elif metric == 'standing_count':
                    data[date_str] = summary.get("standing_count", 0)
                elif metric == 'drinking_count':
                    data[date_str] = summary.get("drinking_count", 0)
        
        # Convert to lists for easier plotting
        dates = list(data.keys())
        values = list(data.values())
        
        # Calculate statistics
        avg_value = np.mean(values) if values else 0
        max_value = max(values) if values else 0
        min_value = min(values) if values else 0
        
        # Calculate trend (positive or negative)
        if len(values) > 1:
            # Simple linear regression
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        else:
            trend_direction = "stable"
        
        return {
            "metric": metric,
            "dates": dates,
            "values": values,
            "avg_value": avg_value,
            "max_value": max_value,
            "min_value": min_value,
            "trend_direction": trend_direction,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
        }
    
    def get_weekly_comparison(
        self, 
        weeks: int = 4
    ) -> Dict[str, Any]:
        """Get weekly comparison data.
        
        Args:
            weeks: Number of weeks to include in the comparison
            
        Returns:
            Dictionary with weekly comparison data
        """
        # Calculate date range
        today = date.today()
        # Get the most recent Monday (0 = Monday in weekday())
        current_week_start = today - timedelta(days=today.weekday())
        
        weekly_data = []
        
        # Get data for each week
        for i in range(weeks):
            week_start = current_week_start - timedelta(days=7*i)
            week_end = week_start + timedelta(days=6)
            
            # Get weekly summary
            weekly_summary = self.db.get_weekly_summary(week_start)
            
            weekly_data.append({
                "week_start": week_start.strftime('%Y-%m-%d'),
                "week_end": week_end.strftime('%Y-%m-%d'),
                "total_standing_time": weekly_summary.get("total_standing_time", 0),
                "total_standing_count": weekly_summary.get("total_standing_count", 0),
                "total_drinking_count": weekly_summary.get("total_drinking_count", 0),
                "avg_standing_time_per_day": weekly_summary.get("avg_standing_time_per_day", 0),
                "avg_standing_count_per_day": weekly_summary.get("avg_standing_count_per_day", 0),
                "avg_drinking_count_per_day": weekly_summary.get("avg_drinking_count_per_day", 0),
            })
        
        # Calculate week-over-week changes
        for i in range(1, len(weekly_data)):
            current = weekly_data[i-1]
            previous = weekly_data[i]
            
            # Calculate percent changes
            for metric in ["total_standing_time", "total_standing_count", "total_drinking_count"]:
                if previous[metric] > 0:
                    change = ((current[metric] - previous[metric]) / previous[metric]) * 100
                    current[f"{metric}_change"] = change
                else:
                    current[f"{metric}_change"] = 0
        
        return {
            "weeks": weeks,
            "weekly_data": weekly_data,
        }
    
    def get_health_score(
        self, 
        date_value: Optional[date] = None
    ) -> Dict[str, Any]:
        """Calculate an overall health score based on standing, drinking, and heart rate.
        
        Args:
            date_value: Date to calculate score for (defaults to today)
            
        Returns:
            Dictionary with health score and component scores
        """
        # Use today if no date provided
        if date_value is None:
            date_value = date.today()
        
        # Get daily summary
        summaries = self.db.get_daily_summaries(date_value, date_value)
        
        if not summaries:
            return {
                "date": date_value.strftime('%Y-%m-%d'),
                "overall_score": 0,
                "standing_score": 0,
                "drinking_score": 0,
                "heart_rate_score": 0,
                "standing_time": 0,
                "standing_count": 0,
                "drinking_count": 0,
                "avg_heart_rate": 0,
            }
        
        summary = summaries[0]
        
        # Get metrics
        standing_time = summary.get("total_standing_time", 0)
        standing_count = summary.get("standing_count", 0)
        drinking_count = summary.get("drinking_count", 0)
        avg_heart_rate = summary.get("avg_heart_rate", 0)
        
        # Calculate component scores
        
        # Standing score (0-100)
        # Target: 2 hours (7200 seconds) of standing per day
        standing_time_score = min(100, (standing_time / 7200) * 100)
        
        # Standing frequency score (0-100)
        # Target: 8 standing sessions per day
        standing_frequency_score = min(100, (standing_count / 8) * 100)
        
        # Combined standing score
        standing_score = (standing_time_score * 0.7) + (standing_frequency_score * 0.3)
        
        # Drinking score (0-100)
        # Target: 8 drinking events per day
        drinking_score = min(100, (drinking_count / 8) * 100)
        
        # Heart rate score (0-100)
        # Target: Heart rate within healthy range (60-100 BPM)
        heart_rate_score = 0
        if avg_heart_rate > 0:
            # Get user settings for heart rate range
            settings = self.db.get_setting("user_settings", {})
            min_heart_rate = settings.get("heart_rate_min", 60)
            max_heart_rate = settings.get("heart_rate_max", 100)
            
            # Calculate score based on how close heart rate is to the middle of the healthy range
            if min_heart_rate <= avg_heart_rate <= max_heart_rate:
                heart_rate_score = 100  # Within healthy range
            else:
                # Calculate how far outside the range
                if avg_heart_rate < min_heart_rate:
                    distance = min_heart_rate - avg_heart_rate
                    max_distance = min_heart_rate * 0.5  # 50% below min is max penalty
                else:  # avg_heart_rate > max_heart_rate
                    distance = avg_heart_rate - max_heart_rate
                    max_distance = max_heart_rate * 0.5  # 50% above max is max penalty
                
                # Convert to score (100 = perfect, 0 = max penalty)
                heart_rate_score = max(0, 100 - (distance / max_distance) * 100)
        
        # Overall health score (0-100)
        # 60% standing, 20% drinking, 20% heart rate
        if avg_heart_rate > 0:
            overall_score = (standing_score * 0.6) + (drinking_score * 0.2) + (heart_rate_score * 0.2)
        else:
            # If no heart rate data, use original formula
            overall_score = (standing_score * 0.7) + (drinking_score * 0.3)
        
        return {
            "date": date_value.strftime('%Y-%m-%d'),
            "overall_score": overall_score,
            "standing_score": standing_score,
            "drinking_score": drinking_score,
            "heart_rate_score": heart_rate_score,
            "standing_time": standing_time,
            "standing_count": standing_count,
            "drinking_count": drinking_count,
            "avg_heart_rate": avg_heart_rate,
        }
    
    def get_recommendations(
        self, 
        date_value: Optional[date] = None
    ) -> List[str]:
        """Generate health recommendations based on recent activity.
        
        Args:
            date_value: Date to generate recommendations for (defaults to today)
            
        Returns:
            List of recommendation strings
        """
        # Use today if no date provided
        if date_value is None:
            date_value = date.today()
        
        # Get health score
        health_score = self.get_health_score(date_value)
        
        recommendations = []
        
        # Standing recommendations
        standing_time = health_score["standing_time"]
        standing_count = health_score["standing_count"]
        
        if standing_time < 3600:  # Less than 1 hour
            recommendations.append(
                "Try to stand for at least 1 hour per day. Consider using a standing desk or taking standing breaks."
            )
        elif standing_time < 7200:  # Less than 2 hours
            recommendations.append(
                "You're doing well with standing, but aim for 2 hours per day for optimal health benefits."
            )
        
        if standing_count < 4:  # Less than 4 standing sessions
            recommendations.append(
                "Try to stand up more frequently. Aim for at least 8 standing sessions per day, even if they're short."
            )
        
        # Drinking recommendations
        drinking_count = health_score["drinking_count"]
        
        if drinking_count < 4:  # Less than 4 drinking events
            recommendations.append(
                "Try to drink water more frequently. Aim for at least 8 glasses of water per day."
            )
        elif drinking_count < 6:  # Less than 6 drinking events
            recommendations.append(
                "You're doing well with hydration, but try to increase to 8 glasses of water per day."
            )
        
        # Heart rate recommendations
        avg_heart_rate = health_score["avg_heart_rate"]
        
        if avg_heart_rate > 0:  # If heart rate data exists
            # Get user settings for heart rate range
            settings = self.db.get_setting("user_settings", {})
            min_heart_rate = settings.get("heart_rate_min", 60)
            max_heart_rate = settings.get("heart_rate_max", 100)
            
            if avg_heart_rate < min_heart_rate:
                recommendations.append(
                    f"Your average heart rate ({avg_heart_rate:.1f} BPM) is below the recommended range. "
                    "Consider moderate exercise to improve cardiovascular health."
                )
            elif avg_heart_rate > max_heart_rate:
                recommendations.append(
                    f"Your average heart rate ({avg_heart_rate:.1f} BPM) is above the recommended range. "
                    "Consider relaxation techniques and consult a healthcare professional if it persists."
                )
            else:
                recommendations.append(
                    f"Your heart rate is within a healthy range at {avg_heart_rate:.1f} BPM. Keep up the good work!"
                )
        
        # Add a general recommendation if doing well
        if health_score["overall_score"] >= 80:
            recommendations.append(
                "Great job maintaining healthy habits! Keep up the good work."
            )
        
        # If no specific recommendations, add a general one
        if not recommendations:
            recommendations.append(
                "Maintain a balance of standing and sitting throughout the day, and stay hydrated."
            )
        
        return recommendations
    
    def close(self) -> None:
        """Close the database connection."""
        if self.db:
            self.db.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create analytics engine
    with HealthAnalytics() as analytics:
        # Get standing statistics
        standing_stats = analytics.get_standing_stats()
        print(f"Standing stats: {standing_stats}")
        
        # Get drinking statistics
        drinking_stats = analytics.get_drinking_stats()
        print(f"Drinking stats: {drinking_stats}")
        
        # Get daily trend
        standing_trend = analytics.get_daily_trend('standing_time', days=7)
        print(f"Standing trend: {standing_trend}")
        
        # Get health score
        health_score = analytics.get_health_score()
        print(f"Health score: {health_score}")
        
        # Get recommendations
        recommendations = analytics.get_recommendations()
        print(f"Recommendations: {recommendations}")
