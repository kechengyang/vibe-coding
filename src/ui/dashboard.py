"""
Dashboard Module for Employee Health Monitoring System.

This module defines the dashboard UI component for displaying health statistics.
"""
import logging
from datetime import date, timedelta
from typing import Dict, List, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFrame, QScrollArea, QSizePolicy, QGridLayout, QSpacerItem,
    QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from ..data.analytics import HealthAnalytics
from ..data.models import DailySummary, HealthScore, PostureEvent, DrinkingEvent
from ..utils.config import get_config
from .activity_log import ActivityLog

logger = logging.getLogger("employee_health_monitor.ui.dashboard")

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


class StatCard(QFrame):
    """Widget for displaying a single statistic."""
    
    def __init__(self, title: str, value: str, subtitle: str = "", parent=None):
        """Initialize the stat card.
        
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
            StatCard {
                background-color: #ffffff;
                border-radius: 12px;
                border: 2px solid #0d6efd;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        
        # Create title label
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #0d6efd; font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Create value label
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #212529; font-size: 32px; font-weight: bold;")
        layout.addWidget(self.value_label)
        
        # Create subtitle label if provided
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("color: #495057; font-size: 12px; background-color: #e9ecef; padding: 2px 4px; border-radius: 2px;")
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


class RecommendationCard(QFrame):
    """Widget for displaying health recommendations."""
    
    def __init__(self, recommendations: List[str] = None, parent=None):
        """Initialize the recommendation card.
        
        Args:
            recommendations: List of recommendation strings
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up frame style
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet("""
            RecommendationCard {
                background-color: #ffffff;
                border-radius: 12px;
                border: 2px solid #198754;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        """)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(12)
        
        # Create title label with icon
        title_layout = QHBoxLayout()
        title_layout.setSpacing(8)
        
        title_label = QLabel("Recommendations")
        title_label.setStyleSheet("color: #198754; font-size: 18px; font-weight: bold;")
        title_layout.addWidget(title_label)
        
        self.layout.addLayout(title_layout)
        
        # Create recommendations list
        self.recommendations_layout = QVBoxLayout()
        self.layout.addLayout(self.recommendations_layout)
        
        # Add recommendations if provided
        if recommendations:
            self.update_recommendations(recommendations)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(150)
    
    def update_recommendations(self, recommendations: List[str]) -> None:
        """Update the displayed recommendations.
        
        Args:
            recommendations: List of recommendation strings
        """
        # Clear existing recommendations
        for i in reversed(range(self.recommendations_layout.count())):
            widget = self.recommendations_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Add new recommendations
        for recommendation in recommendations:
            label = QLabel(f"• {recommendation}")
            label.setWordWrap(True)
            label.setStyleSheet("color: #212529; font-size: 14px; margin-top: 8px; line-height: 1.4; background-color: #f8f9fa; padding: 8px; border-radius: 4px; border-left: 4px solid #198754;")
            self.recommendations_layout.addWidget(label)


class Dashboard(QWidget):
    """Dashboard widget for displaying health statistics."""
    
    def __init__(self, parent=None):
        """Initialize the dashboard.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create analytics engine
        self.analytics = HealthAnalytics()
        
        # Create activity log
        self.activity_log = ActivityLog()
        
        # Set up UI
        self.init_ui()
        
        # Set up refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
        
        # Initial data load
        self.refresh_data()
        
        logger.info("Dashboard initialized")
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Create header
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("Health Dashboard")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0d6efd;")
        header_layout.addWidget(title_label)
        
        # Time period selector
        header_layout.addStretch()
        period_label = QLabel("Time Period:")
        period_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")
        header_layout.addWidget(period_label)
        
        self.period_combo = QComboBox()
        self.period_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 2px solid #0d6efd;
                border-radius: 4px;
                padding: 5px 10px;
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
        self.period_combo.addItems(["Today", "Yesterday", "Last 7 Days", "Last 30 Days"])
        self.period_combo.setCurrentIndex(0)
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        header_layout.addWidget(self.period_combo)
        
        main_layout.addLayout(header_layout)
        
        # Create splitter for main content and activity log
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #0d6efd;
                width: 2px;
                margin: 5px 5px;
            }
            QSplitter::handle:hover {
                background-color: #0b5ed7;
                width: 3px;
            }
        """)
        
        # Create scroll area for dashboard content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #f8f9fa;
                border: none;
            }
        """)
        
        # Create content widget
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #f8f9fa;")
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(20)
        
        # Add stat cards
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        
        self.standing_time_card = StatCard("Standing Time", "0 min", "Total time spent standing")
        stats_layout.addWidget(self.standing_time_card)
        
        self.standing_count_card = StatCard("Standing Sessions", "0", "Number of times stood up")
        stats_layout.addWidget(self.standing_count_card)
        
        self.drinking_count_card = StatCard("Water Intake", "0", "Number of times drank water")
        stats_layout.addWidget(self.drinking_count_card)
        
        self.health_score_card = StatCard("Health Score", "0%", "Overall health score")
        stats_layout.addWidget(self.health_score_card)
        
        self.content_layout.addLayout(stats_layout)
        
        # Add recommendations card
        self.recommendations_card = RecommendationCard()
        self.content_layout.addWidget(self.recommendations_card)
        
        # Add charts
        charts_layout = QGridLayout()
        charts_layout.setColumnStretch(0, 1)
        charts_layout.setColumnStretch(1, 1)
        charts_layout.setSpacing(15)
        
        # Standing time chart
        self.standing_time_chart = MplCanvas(width=5, height=3)
        charts_layout.addWidget(self.standing_time_chart, 0, 0)
        
        # Drinking count chart
        self.drinking_count_chart = MplCanvas(width=5, height=3)
        charts_layout.addWidget(self.drinking_count_chart, 0, 1)
        
        # Health score chart
        self.health_score_chart = MplCanvas(width=10, height=3)
        charts_layout.addWidget(self.health_score_chart, 1, 0, 1, 2)
        
        self.content_layout.addLayout(charts_layout)
        
        # Add spacer at the bottom
        self.content_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Set scroll area widget
        scroll_area.setWidget(content_widget)
        
        # Create activity log container
        activity_log_container = QWidget()
        activity_log_layout = QVBoxLayout(activity_log_container)
        activity_log_layout.setContentsMargins(10, 0, 10, 0)
        activity_log_layout.addWidget(self.activity_log)
        
        # Add widgets to splitter
        splitter.addWidget(scroll_area)
        splitter.addWidget(activity_log_container)
        
        # Set initial sizes (70% for dashboard, 30% for activity log)
        splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
    
    @pyqtSlot()
    def refresh_data(self) -> None:
        """Refresh dashboard data."""
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
            
            # Get standing statistics
            standing_stats = self.analytics.get_standing_stats(start_date, end_date)
            
            # Get drinking statistics
            drinking_stats = self.analytics.get_drinking_stats(start_date, end_date)
            
            # Get health score
            health_score = self.analytics.get_health_score(today)
            
            # Get recommendations
            recommendations = self.analytics.get_recommendations(today)
            
            # Update stat cards
            self.standing_time_card.update_value(f"{int(standing_stats['total_duration'] / 60)} min")
            self.standing_count_card.update_value(str(standing_stats['total_events']))
            self.drinking_count_card.update_value(str(drinking_stats['total_events']))
            self.health_score_card.update_value(f"{int(health_score['overall_score'])}%")
            
            # Update recommendations
            self.recommendations_card.update_recommendations(recommendations)
            
            # Update charts
            self.update_charts(start_date, end_date)
            
            logger.info(f"Dashboard data refreshed for period: {period}")
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard data: {str(e)}")
    
    def update_charts(self, start_date: date, end_date: date) -> None:
        """Update dashboard charts.
        
        Args:
            start_date: Start date for chart data
            end_date: End date for chart data
        """
        try:
            # Get trend data
            standing_trend = self.analytics.get_daily_trend('standing_time', (end_date - start_date).days + 1)
            drinking_trend = self.analytics.get_daily_trend('drinking_count', (end_date - start_date).days + 1)
            
            # Update standing time chart
            self.standing_time_chart.axes.clear()
            
            # Set background color for better contrast
            self.standing_time_chart.fig.patch.set_facecolor('#f8f9fa')
            self.standing_time_chart.axes.set_facecolor('#f8f9fa')
            
            dates = [d.split('-')[2] for d in standing_trend['dates']]  # Just day part
            values = [v / 60 for v in standing_trend['values']]  # Convert to minutes
            
            # Create bars with better styling
            bars = self.standing_time_chart.axes.bar(
                dates, 
                values, 
                color='#0d6efd', 
                edgecolor='white',
                linewidth=1,
                alpha=0.8
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    self.standing_time_chart.axes.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 1,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='#495057'
                    )
            
            # Improved styling
            self.standing_time_chart.axes.set_title('Standing Time (minutes)', fontsize=12, fontweight='bold', color='#0d6efd')
            self.standing_time_chart.axes.set_xlabel('Day', fontsize=10, fontweight='bold', color='#495057')
            self.standing_time_chart.axes.set_ylabel('Minutes', fontsize=10, fontweight='bold', color='#495057')
            self.standing_time_chart.axes.grid(True, linestyle='--', alpha=0.7, color='#dee2e6')
            self.standing_time_chart.axes.tick_params(axis='both', colors='#495057')
            self.standing_time_chart.axes.spines['top'].set_visible(False)
            self.standing_time_chart.axes.spines['right'].set_visible(False)
            
            self.standing_time_chart.fig.tight_layout()
            self.standing_time_chart.draw()
            
            # Update drinking count chart
            self.drinking_count_chart.axes.clear()
            
            # Set background color for better contrast
            self.drinking_count_chart.fig.patch.set_facecolor('#f8f9fa')
            self.drinking_count_chart.axes.set_facecolor('#f8f9fa')
            
            dates = [d.split('-')[2] for d in drinking_trend['dates']]  # Just day part
            values = drinking_trend['values']
            
            # Create bars with better styling
            bars = self.drinking_count_chart.axes.bar(
                dates, 
                values, 
                color='#198754', 
                edgecolor='white',
                linewidth=1,
                alpha=0.8
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    self.drinking_count_chart.axes.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='#495057'
                    )
            
            # Improved styling
            self.drinking_count_chart.axes.set_title('Water Intake (count)', fontsize=12, fontweight='bold', color='#198754')
            self.drinking_count_chart.axes.set_xlabel('Day', fontsize=10, fontweight='bold', color='#495057')
            self.drinking_count_chart.axes.set_ylabel('Count', fontsize=10, fontweight='bold', color='#495057')
            self.drinking_count_chart.axes.grid(True, linestyle='--', alpha=0.7, color='#dee2e6')
            self.drinking_count_chart.axes.tick_params(axis='both', colors='#495057')
            self.drinking_count_chart.axes.spines['top'].set_visible(False)
            self.drinking_count_chart.axes.spines['right'].set_visible(False)
            
            self.drinking_count_chart.fig.tight_layout()
            self.drinking_count_chart.draw()
            
            # Update health score chart
            self.health_score_chart.axes.clear()
            
            # Set background color for better contrast
            self.health_score_chart.fig.patch.set_facecolor('#f8f9fa')
            self.health_score_chart.axes.set_facecolor('#f8f9fa')
            
            # Get health scores for date range
            scores = []
            dates = []
            
            current_date = start_date
            while current_date <= end_date:
                score = self.analytics.get_health_score(current_date)
                scores.append(score['overall_score'])
                dates.append(current_date.strftime('%d'))
                current_date += timedelta(days=1)
            
            # Plot with improved styling
            line = self.health_score_chart.axes.plot(
                dates, 
                scores, 
                marker='o', 
                linestyle='-', 
                color='#dc3545', 
                linewidth=2.5,
                markersize=8
            )[0]
            
            # Add value labels on data points
            for i, (x, y) in enumerate(zip(dates, scores)):
                self.health_score_chart.axes.text(
                    x, y + 3,
                    f'{int(y)}%',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='#495057'
                )
            
            # Fill area under the line
            self.health_score_chart.axes.fill_between(
                dates, 
                scores, 
                alpha=0.2, 
                color='#dc3545'
            )
            
            # Improved styling
            self.health_score_chart.axes.set_title('Health Score Trend', fontsize=14, fontweight='bold', color='#dc3545')
            self.health_score_chart.axes.set_xlabel('Day', fontsize=10, fontweight='bold', color='#495057')
            self.health_score_chart.axes.set_ylabel('Score (%)', fontsize=10, fontweight='bold', color='#495057')
            self.health_score_chart.axes.set_ylim(0, 100)
            self.health_score_chart.axes.grid(True, linestyle='--', alpha=0.7, color='#dee2e6')
            self.health_score_chart.axes.tick_params(axis='both', colors='#495057')
            self.health_score_chart.axes.spines['top'].set_visible(False)
            self.health_score_chart.axes.spines['right'].set_visible(False)
            
            self.health_score_chart.fig.tight_layout()
            self.health_score_chart.draw()
            
            logger.info("Dashboard charts updated")
            
        except Exception as e:
            logger.error(f"Error updating dashboard charts: {str(e)}")
    
    @pyqtSlot(int)
    def on_period_changed(self, index: int) -> None:
        """Handle period selection change.
        
        Args:
            index: Selected index in the combo box
        """
        self.refresh_data()
    
    @pyqtSlot(object)
    def on_posture_event(self, event: PostureEvent) -> None:
        """Handle posture event.
        
        Args:
            event: Posture event
        """
        try:
            # Add event to activity log
            self.activity_log.add_posture_event(event)
            
            # Update dashboard stats in real-time if event is STANDING
            if event.state.name == "STANDING":
                # Get today's date
                today = date.today()
                
                # Get updated standing statistics
                standing_stats = self.analytics.get_standing_stats(today, today)
                
                # Update standing time card
                self.standing_time_card.update_value(f"{int(standing_stats['total_duration'] / 60)} min")
                
                # Update standing count card
                self.standing_count_card.update_value(str(standing_stats['total_events']))
                
                # Get updated health score
                health_score = self.analytics.get_health_score(today)
                
                # Update health score card
                self.health_score_card.update_value(f"{int(health_score['overall_score'])}%")
                
                # Get updated recommendations
                recommendations = self.analytics.get_recommendations(today)
                
                # Update recommendations
                self.recommendations_card.update_recommendations(recommendations)
                
                # Update charts if current period is Today
                if self.period_combo.currentText() == "Today":
                    self.update_charts(today, today)
                
                logger.info("Dashboard updated in real-time for posture event")
                
        except Exception as e:
            logger.error(f"Error updating dashboard for posture event: {str(e)}")
    
    @pyqtSlot(object)
    def on_drinking_event(self, event: DrinkingEvent) -> None:
        """Handle drinking event.
        
        Args:
            event: Drinking event
        """
        try:
            # Add event to activity log
            self.activity_log.add_drinking_event(event)
            
            # Update dashboard stats in real-time
            # Get today's date
            today = date.today()
            
            # Get updated drinking statistics
            drinking_stats = self.analytics.get_drinking_stats(today, today)
            
            # Update drinking count card
            self.drinking_count_card.update_value(str(drinking_stats['total_events']))
            
            # Get updated health score
            health_score = self.analytics.get_health_score(today)
            
            # Update health score card
            self.health_score_card.update_value(f"{int(health_score['overall_score'])}%")
            
            # Get updated recommendations
            recommendations = self.analytics.get_recommendations(today)
            
            # Update recommendations
            self.recommendations_card.update_recommendations(recommendations)
            
            # Update charts if current period is Today
            if self.period_combo.currentText() == "Today":
                self.update_charts(today, today)
            
            logger.info("Dashboard updated in real-time for drinking event")
            
        except Exception as e:
            logger.error(f"Error updating dashboard for drinking event: {str(e)}")
    
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
    
    dashboard = Dashboard()
    dashboard.setWindowTitle("Health Dashboard")
    dashboard.resize(800, 600)
    dashboard.show()
    
    sys.exit(app.exec())
