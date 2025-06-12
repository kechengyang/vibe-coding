"""
Heart Rate View Module for Employee Health Monitoring System.

This module defines the heart rate view UI component for displaying heart rate data.
"""
import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFrame, QSizePolicy, QGridLayout, QSpacerItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from ..data.analytics import HealthAnalytics
from ..utils.logger import setup_logger

# Set up logger
logger = setup_logger("employee_health_monitor.ui.heart_rate_view")

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt."""
    
    def __init__(self, width=5, height=4, dpi=100):
        """Initialize the canvas.
        
        Args:
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


class HeartRateCard(QFrame):
    """Widget for displaying heart rate statistics."""
    
    def __init__(self, title: str, value: str, subtitle: str = "", parent=None):
        """Initialize the heart rate card.
        
        Args:
            title: Card title
            value: Main value to display
            subtitle: Optional subtitle or description
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up frame style
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet("""
            HeartRateCard {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create title label
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #6c757d; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Create value label
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #212529; font-size: 24px; font-weight: bold;")
        layout.addWidget(self.value_label)
        
        # Create subtitle label if provided
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("color: #6c757d; font-size: 12px;")
            layout.addWidget(subtitle_label)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(100)
    
    def update_value(self, value: str) -> None:
        """Update the displayed value.
        
        Args:
            value: New value to display
        """
        self.value_label.setText(value)


class HeartRateView(QWidget):
    """Widget for displaying heart rate data and statistics."""
    
    def __init__(self, parent=None):
        """Initialize the heart rate view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create analytics engine
        self.analytics = HealthAnalytics()
        
        # Set up UI
        self.init_ui()
        
        # Set up refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # Initial data load
        self.refresh_data()
        
        logger.info("Heart rate view initialized")
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Create header
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("Heart Rate Monitor")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #212529;")
        header_layout.addWidget(title_label)
        
        # Time period selector
        header_layout.addStretch()
        period_label = QLabel("Time Period:")
        period_label.setStyleSheet("font-size: 14px; color: #6c757d;")
        header_layout.addWidget(period_label)
        
        self.period_combo = QComboBox()
        self.period_combo.addItems(["Today", "Yesterday", "Last 7 Days", "Last 30 Days"])
        self.period_combo.setCurrentIndex(0)
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        header_layout.addWidget(self.period_combo)
        
        main_layout.addLayout(header_layout)
        
        # Create stat cards
        stats_layout = QHBoxLayout()
        
        self.current_bpm_card = HeartRateCard("Current BPM", "-- BPM", "Current heart rate")
        stats_layout.addWidget(self.current_bpm_card)
        
        self.avg_bpm_card = HeartRateCard("Average BPM", "-- BPM", "Average heart rate")
        stats_layout.addWidget(self.avg_bpm_card)
        
        self.min_bpm_card = HeartRateCard("Minimum BPM", "-- BPM", "Minimum heart rate")
        stats_layout.addWidget(self.min_bpm_card)
        
        self.max_bpm_card = HeartRateCard("Maximum BPM", "-- BPM", "Maximum heart rate")
        stats_layout.addWidget(self.max_bpm_card)
        
        main_layout.addLayout(stats_layout)
        
        # Create charts
        charts_layout = QVBoxLayout()
        
        # Heart rate trend chart
        self.trend_chart = MplCanvas(width=10, height=4)
        charts_layout.addWidget(self.trend_chart)
        
        # Heart rate distribution chart
        self.distribution_chart = MplCanvas(width=10, height=4)
        charts_layout.addWidget(self.distribution_chart)
        
        main_layout.addLayout(charts_layout)
        
        # Add spacer at the bottom
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
    
    @pyqtSlot()
    def refresh_data(self) -> None:
        """Refresh heart rate data."""
        try:
            # Get selected time period
            period = self.period_combo.currentText()
            
            # Calculate date range
            today = date.today()
            
            if period == "Today":
                start_date = today
                end_date = today
            elif period == "Yesterday":
                start_date = today - timedelta(days=1)
                end_date = start_date
            elif period == "Last 7 Days":
                start_date = today - timedelta(days=6)
                end_date = today
            elif period == "Last 30 Days":
                start_date = today - timedelta(days=29)
                end_date = today
            else:
                start_date = today
                end_date = today
            
            # Get heart rate statistics
            heart_rate_stats = self.analytics.get_heart_rate_stats(start_date, end_date)
            
            # Get heart rate events
            heart_rate_events = self.analytics.db.get_heart_rate_events(start_date, end_date)
            
            # Update stat cards
            if heart_rate_events:
                # Use the most recent event for current BPM
                current_bpm = heart_rate_events[0]["bpm"]
                self.current_bpm_card.update_value(f"{current_bpm:.1f} BPM")
            else:
                self.current_bpm_card.update_value("-- BPM")
            
            self.avg_bpm_card.update_value(f"{heart_rate_stats['avg_bpm']:.1f} BPM" if heart_rate_stats['avg_bpm'] > 0 else "-- BPM")
            self.min_bpm_card.update_value(f"{heart_rate_stats['min_bpm']:.1f} BPM" if heart_rate_stats['min_bpm'] > 0 else "-- BPM")
            self.max_bpm_card.update_value(f"{heart_rate_stats['max_bpm']:.1f} BPM" if heart_rate_stats['max_bpm'] > 0 else "-- BPM")
            
            # Update charts
            self.update_charts(heart_rate_events)
            
            logger.info(f"Heart rate data refreshed for period: {period}")
            
        except Exception as e:
            logger.error(f"Error refreshing heart rate data: {str(e)}")
    
    def update_charts(self, heart_rate_events: List[Dict[str, Any]]) -> None:
        """Update heart rate charts.
        
        Args:
            heart_rate_events: List of heart rate events
        """
        try:
            # Update trend chart
            self.trend_chart.axes.clear()
            
            if heart_rate_events:
                # Sort events by timestamp (oldest first)
                sorted_events = sorted(heart_rate_events, key=lambda x: x["timestamp"])
                
                # Extract timestamps and BPM values
                timestamps = [datetime.fromtimestamp(event["timestamp"]) for event in sorted_events]
                bpm_values = [event["bpm"] for event in sorted_events]
                confidence_values = [event["confidence"] for event in sorted_events]
                
                # Plot heart rate trend
                self.trend_chart.axes.plot(timestamps, bpm_values, marker='o', linestyle='-', color='#e63946')
                
                # Set y-axis range with some padding
                min_bpm = min(bpm_values) if bpm_values else 0
                max_bpm = max(bpm_values) if bpm_values else 100
                padding = (max_bpm - min_bpm) * 0.1 if bpm_values else 10
                self.trend_chart.axes.set_ylim(max(0, min_bpm - padding), max_bpm + padding)
                
                # Format x-axis to show time
                self.trend_chart.fig.autofmt_xdate()
                
                # Add confidence as alpha to scatter points
                for i, (timestamp, bpm, confidence) in enumerate(zip(timestamps, bpm_values, confidence_values)):
                    self.trend_chart.axes.scatter(
                        timestamp, bpm, 
                        alpha=confidence, 
                        color='#e63946', 
                        s=50
                    )
            
            self.trend_chart.axes.set_title('Heart Rate Trend')
            self.trend_chart.axes.set_xlabel('Time')
            self.trend_chart.axes.set_ylabel('BPM')
            self.trend_chart.axes.grid(True, linestyle='--', alpha=0.7)
            self.trend_chart.fig.tight_layout()
            self.trend_chart.draw()
            
            # Update distribution chart
            self.distribution_chart.axes.clear()
            
            if heart_rate_events:
                # Extract BPM values
                bpm_values = [event["bpm"] for event in heart_rate_events]
                
                # Create histogram
                self.distribution_chart.axes.hist(
                    bpm_values, 
                    bins=20, 
                    color='#457b9d', 
                    alpha=0.7, 
                    edgecolor='black'
                )
                
                # Add vertical lines for min, max, and average
                avg_bpm = np.mean(bpm_values)
                min_bpm = min(bpm_values)
                max_bpm = max(bpm_values)
                
                self.distribution_chart.axes.axvline(avg_bpm, color='#e63946', linestyle='--', linewidth=2, label=f'Avg: {avg_bpm:.1f}')
                self.distribution_chart.axes.axvline(min_bpm, color='#2a9d8f', linestyle=':', linewidth=2, label=f'Min: {min_bpm:.1f}')
                self.distribution_chart.axes.axvline(max_bpm, color='#f4a261', linestyle=':', linewidth=2, label=f'Max: {max_bpm:.1f}')
                
                # Add legend
                self.distribution_chart.axes.legend()
            
            self.distribution_chart.axes.set_title('Heart Rate Distribution')
            self.distribution_chart.axes.set_xlabel('BPM')
            self.distribution_chart.axes.set_ylabel('Frequency')
            self.distribution_chart.axes.grid(True, linestyle='--', alpha=0.7)
            self.distribution_chart.fig.tight_layout()
            self.distribution_chart.draw()
            
            logger.info("Heart rate charts updated")
            
        except Exception as e:
            logger.error(f"Error updating heart rate charts: {str(e)}")
    
    @pyqtSlot(int)
    def on_period_changed(self, index: int) -> None:
        """Handle period selection change.
        
        Args:
            index: Selected index in the combo box
        """
        self.refresh_data()
    
    def closeEvent(self, event) -> None:
        """Handle widget close event."""
        # Stop refresh timer
        self.refresh_timer.stop()
        
        # Close analytics
        if self.analytics:
            self.analytics.close()
        
        super().closeEvent(event)


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    heart_rate_view = HeartRateView()
    heart_rate_view.setWindowTitle("Heart Rate Monitor")
    heart_rate_view.resize(800, 600)
    heart_rate_view.show()
    
    sys.exit(app.exec())
