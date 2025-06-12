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
                background-color: #ffffff;
                border-radius: 8px;
                border: 2px solid #0d6efd;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(5)
        
        # Create title label
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #0d6efd; font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Create value label
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #212529; font-size: 24px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setWordWrap(True)
        layout.addWidget(self.value_label)
        
        # Create subtitle label if provided
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("color: #495057; font-size: 12px; background-color: #e9ecef; padding: 2px 4px; border-radius: 2px;")
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(subtitle_label)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(120)
        self.setMinimumWidth(150)
    
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
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0d6efd;")
        header_layout.addWidget(title_label)
        
        # Time period selector
        header_layout.addStretch()
        period_label = QLabel("Time Period:")
        period_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")
        header_layout.addWidget(period_label)
        
        self.period_combo = QComboBox()
        self.period_combo.addItems(["Today", "Yesterday", "Last 7 Days", "Last 30 Days"])
        self.period_combo.setCurrentIndex(0)
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        self.period_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 2px solid #0d6efd;
                border-radius: 4px;
                padding: 5px;
                min-width: 150px;
                color: #212529;
                font-weight: bold;
            }
            QComboBox:hover {
                border-color: #0b5ed7;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #0d6efd;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 2px solid #0d6efd;
                selection-background-color: #0d6efd;
                selection-color: white;
            }
        """)
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
        trend_frame = QFrame()
        trend_frame.setFrameShape(QFrame.Shape.StyledPanel)
        trend_frame.setFrameShadow(QFrame.Shadow.Raised)
        trend_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 8px;
                border: 2px solid #0d6efd;
                margin: 5px;
            }
        """)
        trend_layout = QVBoxLayout(trend_frame)
        
        trend_title = QLabel("Heart Rate Trend")
        trend_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #0d6efd; margin-bottom: 10px;")
        trend_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trend_layout.addWidget(trend_title)
        
        self.trend_chart = MplCanvas(width=10, height=4)
        trend_layout.addWidget(self.trend_chart)
        
        charts_layout.addWidget(trend_frame)
        
        # Heart rate distribution chart
        dist_frame = QFrame()
        dist_frame.setFrameShape(QFrame.Shape.StyledPanel)
        dist_frame.setFrameShadow(QFrame.Shadow.Raised)
        dist_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 8px;
                border: 2px solid #0d6efd;
                margin: 5px;
            }
        """)
        dist_layout = QVBoxLayout(dist_frame)
        
        dist_title = QLabel("Heart Rate Distribution")
        dist_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #0d6efd; margin-bottom: 10px;")
        dist_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dist_layout.addWidget(dist_title)
        
        self.distribution_chart = MplCanvas(width=10, height=4)
        dist_layout.addWidget(self.distribution_chart)
        
        charts_layout.addWidget(dist_frame)
        
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
            
            # Set background color for better contrast
            self.trend_chart.fig.patch.set_facecolor('#f8f9fa')
            self.trend_chart.axes.set_facecolor('#f8f9fa')
            
            if heart_rate_events:
                # Sort events by timestamp (oldest first)
                sorted_events = sorted(heart_rate_events, key=lambda x: x["timestamp"])
                
                # Extract timestamps and BPM values
                timestamps = [datetime.fromtimestamp(event["timestamp"]) for event in sorted_events]
                bpm_values = [event["bpm"] for event in sorted_events]
                confidence_values = [event["confidence"] for event in sorted_events]
                
                # Plot heart rate trend with thicker line for better visibility
                self.trend_chart.axes.plot(timestamps, bpm_values, marker='o', linestyle='-', color='#0d6efd', linewidth=2.5)
                
                # Set y-axis range with some padding
                min_bpm = min(bpm_values) if bpm_values else 0
                max_bpm = max(bpm_values) if bpm_values else 100
                padding = (max_bpm - min_bpm) * 0.1 if bpm_values else 10
                self.trend_chart.axes.set_ylim(max(0, min_bpm - padding), max_bpm + padding)
                
                # Format x-axis to show time
                self.trend_chart.fig.autofmt_xdate()
                
                # Add confidence as alpha to scatter points with larger markers
                for i, (timestamp, bpm, confidence) in enumerate(zip(timestamps, bpm_values, confidence_values)):
                    self.trend_chart.axes.scatter(
                        timestamp, bpm, 
                        alpha=max(0.5, confidence), # Minimum alpha of 0.5 for better visibility
                        color='#0d6efd', 
                        s=80,
                        edgecolor='white',
                        linewidth=1
                    )
            
            # Improved styling for better readability
            self.trend_chart.axes.set_xlabel('Time', fontsize=12, fontweight='bold', color='#495057')
            self.trend_chart.axes.set_ylabel('BPM', fontsize=12, fontweight='bold', color='#495057')
            self.trend_chart.axes.grid(True, linestyle='--', alpha=0.7, color='#dee2e6')
            self.trend_chart.axes.tick_params(axis='both', colors='#495057', labelsize=10)
            
            # Add a light background to the plot area for better contrast
            self.trend_chart.axes.set_axisbelow(True)
            
            self.trend_chart.fig.tight_layout()
            self.trend_chart.draw()
            
            # Update distribution chart
            self.distribution_chart.axes.clear()
            
            # Set background color for better contrast
            self.distribution_chart.fig.patch.set_facecolor('#f8f9fa')
            self.distribution_chart.axes.set_facecolor('#f8f9fa')
            
            if heart_rate_events:
                # Extract BPM values
                bpm_values = [event["bpm"] for event in heart_rate_events]
                
                # Create histogram with more vibrant colors
                self.distribution_chart.axes.hist(
                    bpm_values, 
                    bins=20, 
                    color='#6610f2', 
                    alpha=0.7, 
                    edgecolor='white',
                    linewidth=1.5
                )
                
                # Add vertical lines for min, max, and average with more distinct colors
                avg_bpm = np.mean(bpm_values)
                min_bpm = min(bpm_values)
                max_bpm = max(bpm_values)
                
                self.distribution_chart.axes.axvline(avg_bpm, color='#dc3545', linestyle='-', linewidth=3, label=f'Avg: {avg_bpm:.1f} BPM')
                self.distribution_chart.axes.axvline(min_bpm, color='#198754', linestyle='-', linewidth=3, label=f'Min: {min_bpm:.1f} BPM')
                self.distribution_chart.axes.axvline(max_bpm, color='#fd7e14', linestyle='-', linewidth=3, label=f'Max: {max_bpm:.1f} BPM')
                
                # Add legend with better styling
                legend = self.distribution_chart.axes.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor('#ffffff')
                legend.get_frame().set_edgecolor('#0d6efd')
                legend.get_frame().set_linewidth(2)
            
            # Improved styling for better readability
            self.distribution_chart.axes.set_xlabel('BPM', fontsize=12, fontweight='bold', color='#495057')
            self.distribution_chart.axes.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='#495057')
            self.distribution_chart.axes.grid(True, linestyle='--', alpha=0.7, color='#dee2e6')
            self.distribution_chart.axes.tick_params(axis='both', colors='#495057', labelsize=10)
            
            # Add a light background to the plot area for better contrast
            self.distribution_chart.axes.set_axisbelow(True)
            
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
