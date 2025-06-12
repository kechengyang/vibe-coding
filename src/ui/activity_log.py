"""
Activity Log Module for Employee Health Monitoring System.

This module defines the activity log UI component for displaying recent events.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame, 
    QSizePolicy, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize
from PyQt6.QtGui import QIcon, QPixmap, QColor, QPalette

from ..utils.logger import setup_logger
from ..data.models import PostureEvent, DrinkingEvent

# Set up logger
logger = setup_logger("employee_health_monitor.ui.activity_log")

class ActivityItem(QFrame):
    """Widget for displaying a single activity item."""
    
    def __init__(self, 
                 timestamp: str, 
                 activity_type: str, 
                 description: str, 
                 icon_type: str = None, 
                 parent=None):
        """Initialize the activity item.
        
        Args:
            timestamp: Time of the activity
            activity_type: Type of activity (e.g., "Posture", "Drinking")
            description: Description of the activity
            icon_type: Type of icon to display (e.g., "standing", "sitting", "drinking")
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up frame style
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet("""
            ActivityItem {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #e9ecef;
                margin-bottom: 8px;
                padding: 2px;
            }
        """)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Add icon if provided
        if icon_type:
            icon_label = QLabel()
            icon_label.setFixedSize(32, 32)
            
            # Set icon based on type
            if icon_type == "standing":
                icon_label.setStyleSheet("""
                    background-color: #4CAF50;
                    border-radius: 16px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    line-height: 32px;
                """)
                icon_label.setText("S")
            elif icon_type == "sitting":
                icon_label.setStyleSheet("""
                    background-color: #FFC107;
                    border-radius: 16px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    line-height: 32px;
                """)
                icon_label.setText("S")
            elif icon_type == "transitioning":
                icon_label.setStyleSheet("""
                    background-color: #2196F3;
                    border-radius: 16px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    line-height: 32px;
                """)
                icon_label.setText("T")
            elif icon_type == "drinking":
                icon_label.setStyleSheet("""
                    background-color: #03A9F4;
                    border-radius: 16px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    line-height: 32px;
                """)
                icon_label.setText("D")
            
            layout.addWidget(icon_label)
        
        # Create content layout
        content_layout = QVBoxLayout()
        content_layout.setSpacing(4)
        
        # Create header layout
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        # Create activity type label
        type_label = QLabel(activity_type)
        type_label.setStyleSheet("color: #212529; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(type_label)
        
        # Add spacer
        header_layout.addStretch()
        
        # Create timestamp label
        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: #6c757d; font-size: 12px;")
        header_layout.addWidget(time_label)
        
        content_layout.addLayout(header_layout)
        
        # Create description label
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #495057; font-size: 13px;")
        content_layout.addWidget(desc_label)
        
        layout.addLayout(content_layout)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ActivityLog(QWidget):
    """Widget for displaying activity log."""
    
    def __init__(self, parent=None):
        """Initialize the activity log.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Initialize activity items list
        self.activity_items = []
        self.max_items = 50  # Maximum number of items to display
        
        # Set up UI
        self.init_ui()
        
        logger.info("Activity log initialized")
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)
        
        # Create header
        header_label = QLabel("Activity Log")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529;")
        main_layout.addWidget(header_label)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #f8f9fa;
                border-radius: 8px;
            }
        """)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout.setSpacing(8)
        self.content_layout.addStretch()
        
        # Set scroll area widget
        scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(scroll_area)
    
    @pyqtSlot(object)
    def add_posture_event(self, event: PostureEvent) -> None:
        """Add a posture event to the log.
        
        Args:
            event: Posture event
        """
        try:
            # Create timestamp string
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            
            # Create description based on state
            if event.state.name == "STANDING":
                description = "You are now standing"
                icon_type = "standing"
            elif event.state.name == "SITTING":
                description = "You are now sitting"
                icon_type = "sitting"
            elif event.state.name == "TRANSITIONING":
                description = "You are transitioning between postures"
                icon_type = "transitioning"
            else:
                description = f"Posture changed to {event.state.name}"
                icon_type = None
            
            # Add activity item
            self.add_activity_item(timestamp, "Posture", description, icon_type)
            
        except Exception as e:
            logger.error(f"Error adding posture event to activity log: {str(e)}")
    
    @pyqtSlot(object)
    def add_drinking_event(self, event: DrinkingEvent) -> None:
        """Add a drinking event to the log.
        
        Args:
            event: Drinking event
        """
        try:
            # Create timestamp string
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            
            # Create description
            description = f"Drinking detected (duration: {event.duration:.1f}s)"
            
            # Add activity item
            self.add_activity_item(timestamp, "Drinking", description, "drinking")
            
        except Exception as e:
            logger.error(f"Error adding drinking event to activity log: {str(e)}")
    
    def add_activity_item(self, timestamp: str, activity_type: str, 
                          description: str, icon_type: str = None) -> None:
        """Add an activity item to the log.
        
        Args:
            timestamp: Time of the activity
            activity_type: Type of activity
            description: Description of the activity
            icon_type: Type of icon to display
        """
        try:
            # Create activity item
            item = ActivityItem(timestamp, activity_type, description, icon_type)
            
            # Add to layout at the beginning (most recent first)
            self.content_layout.insertWidget(0, item)
            
            # Add to items list
            self.activity_items.append(item)
            
            # Remove oldest items if exceeding maximum
            if len(self.activity_items) > self.max_items:
                oldest_item = self.activity_items.pop(0)
                self.content_layout.removeWidget(oldest_item)
                oldest_item.deleteLater()
            
        except Exception as e:
            logger.error(f"Error adding activity item to log: {str(e)}")
    
    def clear(self) -> None:
        """Clear all activity items."""
        try:
            # Remove all items
            for item in self.activity_items:
                self.content_layout.removeWidget(item)
                item.deleteLater()
            
            # Clear items list
            self.activity_items = []
            
        except Exception as e:
            logger.error(f"Error clearing activity log: {str(e)}")


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    activity_log = ActivityLog()
    activity_log.setWindowTitle("Activity Log")
    activity_log.resize(400, 600)
    
    # Add some example items
    activity_log.add_activity_item("09:15:30", "Posture", "You are now standing", "standing")
    activity_log.add_activity_item("09:20:45", "Posture", "You are now sitting", "sitting")
    activity_log.add_activity_item("09:25:10", "Drinking", "Drinking detected (duration: 3.5s)", "drinking")
    
    activity_log.show()
    
    sys.exit(app.exec())
