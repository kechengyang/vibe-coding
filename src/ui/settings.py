"""
Settings Module for Employee Health Monitoring System.

This module defines the settings UI component for configuring the application.
"""
import logging
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFrame, QScrollArea, QSizePolicy, QGridLayout, QSpacerItem,
    QPushButton, QSlider, QSpinBox, QCheckBox, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QFont

from ..data.models import UserSettings
from ..utils.config import get_config
from ..utils.notifications import get_notification_manager
from ..vision.heart_rate_detection import HeartRateDetector

logger = logging.getLogger("employee_health_monitor.ui.settings")

class SettingsWidget(QWidget):
    """Widget for configuring application settings."""
    
    # Signal emitted when settings are saved
    settings_saved = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the settings widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Get configuration
        self.config = get_config()
        self.user_settings = self.config.get_user_settings()
        
        # Set up UI
        self.init_ui()
        
        # Load settings
        self.load_settings()
        
        logger.info("Settings widget initialized")
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Create header
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("Settings")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #212529;")
        header_layout.addWidget(title_label)
        
        # Save button
        header_layout.addStretch()
        self.save_button = QPushButton("Save Settings")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:pressed {
                background-color: #0062cc;
            }
        """)
        self.save_button.clicked.connect(self.save_settings)
        header_layout.addWidget(self.save_button)
        
        main_layout.addLayout(header_layout)
        
        # Create scroll area for settings content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create content widget
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(20)
        
        # Add settings groups
        self.add_general_settings()
        self.add_health_goal_settings()
        self.add_heart_rate_settings()
        self.add_notification_settings()
        self.add_camera_settings()
        self.add_detection_settings()
        
        # Add spacer at the bottom
        self.content_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Set scroll area widget
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
    
    def add_general_settings(self) -> None:
        """Add general settings group."""
        group_box = QGroupBox("General Settings")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        form_layout = QFormLayout(group_box)
        form_layout.setContentsMargins(15, 20, 15, 15)
        form_layout.setSpacing(15)
        
        # Start monitoring on launch
        self.start_monitoring_checkbox = QCheckBox("Start monitoring on application launch")
        form_layout.addRow("", self.start_monitoring_checkbox)
        
        self.content_layout.addWidget(group_box)
    
    def add_health_goal_settings(self) -> None:
        """Add health goal settings group."""
        group_box = QGroupBox("Health Goals")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        form_layout = QFormLayout(group_box)
        form_layout.setContentsMargins(15, 20, 15, 15)
        form_layout.setSpacing(15)
        
        # Standing goal
        standing_layout = QHBoxLayout()
        self.standing_goal_spinbox = QSpinBox()
        self.standing_goal_spinbox.setRange(30, 240)
        self.standing_goal_spinbox.setSingleStep(10)
        self.standing_goal_spinbox.setSuffix(" minutes")
        self.standing_goal_spinbox.setMinimumWidth(120)
        standing_layout.addWidget(self.standing_goal_spinbox)
        standing_layout.addStretch()
        form_layout.addRow("Daily standing goal:", standing_layout)
        
        # Drinking goal
        drinking_layout = QHBoxLayout()
        self.drinking_goal_spinbox = QSpinBox()
        self.drinking_goal_spinbox.setRange(1, 20)
        self.drinking_goal_spinbox.setSingleStep(1)
        self.drinking_goal_spinbox.setSuffix(" times")
        self.drinking_goal_spinbox.setMinimumWidth(120)
        drinking_layout.addWidget(self.drinking_goal_spinbox)
        drinking_layout.addStretch()
        form_layout.addRow("Daily water intake goal:", drinking_layout)
        
        self.content_layout.addWidget(group_box)
    
    def add_heart_rate_settings(self) -> None:
        """Add heart rate settings group."""
        group_box = QGroupBox("Heart Rate Settings")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        form_layout = QFormLayout(group_box)
        form_layout.setContentsMargins(15, 20, 15, 15)
        form_layout.setSpacing(15)
        
        # Minimum heart rate
        min_hr_layout = QHBoxLayout()
        self.min_heart_rate_spinbox = QSpinBox()
        self.min_heart_rate_spinbox.setRange(40, 100)
        self.min_heart_rate_spinbox.setSingleStep(1)
        self.min_heart_rate_spinbox.setSuffix(" BPM")
        self.min_heart_rate_spinbox.setMinimumWidth(120)
        min_hr_layout.addWidget(self.min_heart_rate_spinbox)
        min_hr_layout.addStretch()
        form_layout.addRow("Minimum healthy heart rate:", min_hr_layout)
        
        # Maximum heart rate
        max_hr_layout = QHBoxLayout()
        self.max_heart_rate_spinbox = QSpinBox()
        self.max_heart_rate_spinbox.setRange(100, 200)
        self.max_heart_rate_spinbox.setSingleStep(1)
        self.max_heart_rate_spinbox.setSuffix(" BPM")
        self.max_heart_rate_spinbox.setMinimumWidth(120)
        max_hr_layout.addWidget(self.max_heart_rate_spinbox)
        max_hr_layout.addStretch()
        form_layout.addRow("Maximum healthy heart rate:", max_hr_layout)
        
        # Heart rate detection interval
        update_interval_layout = QHBoxLayout()
        self.heart_rate_update_interval_spinbox = QSpinBox()
        self.heart_rate_update_interval_spinbox.setRange(10, 60)
        self.heart_rate_update_interval_spinbox.setSingleStep(5)
        self.heart_rate_update_interval_spinbox.setSuffix(" frames")
        self.heart_rate_update_interval_spinbox.setMinimumWidth(120)
        update_interval_layout.addWidget(self.heart_rate_update_interval_spinbox)
        update_interval_layout.addStretch()
        form_layout.addRow("Update heart rate every:", update_interval_layout)
        
        self.content_layout.addWidget(group_box)
    
    def add_notification_settings(self) -> None:
        """Add notification settings group."""
        group_box = QGroupBox("Notifications")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        form_layout = QFormLayout(group_box)
        form_layout.setContentsMargins(15, 20, 15, 15)
        form_layout.setSpacing(15)
        
        # Enable notifications
        self.notifications_enabled_checkbox = QCheckBox("Enable notifications")
        form_layout.addRow("", self.notifications_enabled_checkbox)
        
        # Standing reminder
        standing_reminder_layout = QHBoxLayout()
        self.standing_reminder_spinbox = QSpinBox()
        self.standing_reminder_spinbox.setRange(15, 120)
        self.standing_reminder_spinbox.setSingleStep(5)
        self.standing_reminder_spinbox.setSuffix(" minutes")
        self.standing_reminder_spinbox.setMinimumWidth(120)
        standing_reminder_layout.addWidget(self.standing_reminder_spinbox)
        standing_reminder_layout.addStretch()
        form_layout.addRow("Remind to stand after sitting for:", standing_reminder_layout)
        
        # Drinking reminder
        drinking_reminder_layout = QHBoxLayout()
        self.drinking_reminder_spinbox = QSpinBox()
        self.drinking_reminder_spinbox.setRange(30, 240)
        self.drinking_reminder_spinbox.setSingleStep(15)
        self.drinking_reminder_spinbox.setSuffix(" minutes")
        self.drinking_reminder_spinbox.setMinimumWidth(120)
        drinking_reminder_layout.addWidget(self.drinking_reminder_spinbox)
        drinking_reminder_layout.addStretch()
        form_layout.addRow("Remind to drink water every:", drinking_reminder_layout)
        
        # Test notification button
        test_button_layout = QHBoxLayout()
        self.test_notification_button = QPushButton("Test Notification")
        self.test_notification_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """)
        self.test_notification_button.clicked.connect(self.test_notification)
        test_button_layout.addWidget(self.test_notification_button)
        test_button_layout.addStretch()
        form_layout.addRow("", test_button_layout)
        
        self.content_layout.addWidget(group_box)
    
    def add_camera_settings(self) -> None:
        """Add camera settings group."""
        group_box = QGroupBox("Camera Settings")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        form_layout = QFormLayout(group_box)
        form_layout.setContentsMargins(15, 20, 15, 15)
        form_layout.setSpacing(15)
        
        # Camera selection
        camera_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Default Camera", "Camera 1", "Camera 2", "Camera 3"])
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addStretch()
        form_layout.addRow("Select camera:", camera_layout)
        
        # Test camera button
        test_button_layout = QHBoxLayout()
        self.test_camera_button = QPushButton("Test Camera")
        self.test_camera_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """)
        self.test_camera_button.clicked.connect(self.test_camera)
        test_button_layout.addWidget(self.test_camera_button)
        test_button_layout.addStretch()
        form_layout.addRow("", test_button_layout)
        
        self.content_layout.addWidget(group_box)
    
    def add_detection_settings(self) -> None:
        """Add advanced detection settings group."""
        group_box = QGroupBox("Detection Settings")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        form_layout = QFormLayout(group_box)
        form_layout.setContentsMargins(15, 20, 15, 15)
        form_layout.setSpacing(15)
        
        # Posture model complexity
        posture_model_layout = QHBoxLayout()
        self.posture_model_combo = QComboBox()
        self.posture_model_combo.addItems(["Lite (Fast)", "Full (Balanced)", "Heavy (Accurate)"])
        self.posture_model_combo.setToolTip("Higher accuracy models require more processing power")
        posture_model_layout.addWidget(self.posture_model_combo)
        posture_model_layout.addStretch()
        form_layout.addRow("Posture detection model:", posture_model_layout)
        
        # Add a help label for posture model
        posture_help_label = QLabel("Higher accuracy models provide better detection but require more processing power.")
        posture_help_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        posture_help_label.setWordWrap(True)
        form_layout.addRow("", posture_help_label)
        
        # Drinking model complexity
        drinking_model_layout = QHBoxLayout()
        self.drinking_model_combo = QComboBox()
        self.drinking_model_combo.addItems(["Lite (Fast)", "Full (Balanced)", "Heavy (Accurate)"])
        self.drinking_model_combo.setToolTip("Higher accuracy models require more processing power")
        drinking_model_layout.addWidget(self.drinking_model_combo)
        drinking_model_layout.addStretch()
        form_layout.addRow("Drinking detection model:", drinking_model_layout)
        
        # Add a help label for drinking model
        drinking_help_label = QLabel("Higher accuracy models provide better detection but require more processing power.")
        drinking_help_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        drinking_help_label.setWordWrap(True)
        form_layout.addRow("", drinking_help_label)
        
        self.content_layout.addWidget(group_box)
    
    def load_settings(self) -> None:
        """Load settings from configuration."""
        try:
            # General settings
            self.start_monitoring_checkbox.setChecked(
                self.user_settings.start_monitoring_on_launch
            )
            
            # Health goal settings
            self.standing_goal_spinbox.setValue(
                self.user_settings.standing_goal_minutes
            )
            self.drinking_goal_spinbox.setValue(
                self.user_settings.drinking_goal_count
            )
            
            # Heart rate settings
            self.min_heart_rate_spinbox.setValue(
                self.user_settings.heart_rate_min
            )
            self.max_heart_rate_spinbox.setValue(
                self.user_settings.heart_rate_max
            )
            # Get heart rate update interval from config
            heart_rate_update_interval = self.config.get("detection.heart_rate.update_interval", 30)
            self.heart_rate_update_interval_spinbox.setValue(heart_rate_update_interval)
            
            # Notification settings
            self.notifications_enabled_checkbox.setChecked(
                self.user_settings.notification_enabled
            )
            self.standing_reminder_spinbox.setValue(
                self.user_settings.standing_reminder_minutes
            )
            self.drinking_reminder_spinbox.setValue(
                self.user_settings.drinking_reminder_minutes
            )
            
            # Camera settings
            camera_id = self.user_settings.camera_id
            self.camera_combo.setCurrentIndex(min(camera_id, self.camera_combo.count() - 1))
            
            # Detection settings
            posture_model_complexity = self.user_settings.posture_model_complexity
            self.posture_model_combo.setCurrentIndex(min(posture_model_complexity, 2))
            
            drinking_model_complexity = self.user_settings.drinking_model_complexity
            self.drinking_model_combo.setCurrentIndex(min(drinking_model_complexity, 2))
            
            logger.info("Settings loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
    
    @pyqtSlot()
    def save_settings(self) -> None:
        """Save settings to configuration."""
        try:
            # Update user settings
            self.user_settings.start_monitoring_on_launch = self.start_monitoring_checkbox.isChecked()
            self.user_settings.standing_goal_minutes = self.standing_goal_spinbox.value()
            self.user_settings.drinking_goal_count = self.drinking_goal_spinbox.value()
            self.user_settings.heart_rate_min = self.min_heart_rate_spinbox.value()
            self.user_settings.heart_rate_max = self.max_heart_rate_spinbox.value()
            self.user_settings.notification_enabled = self.notifications_enabled_checkbox.isChecked()
            self.user_settings.standing_reminder_minutes = self.standing_reminder_spinbox.value()
            self.user_settings.drinking_reminder_minutes = self.drinking_reminder_spinbox.value()
            self.user_settings.camera_id = self.camera_combo.currentIndex()
            self.user_settings.posture_model_complexity = self.posture_model_combo.currentIndex()
            self.user_settings.drinking_model_complexity = self.drinking_model_combo.currentIndex()
            
            # Update heart rate detection settings
            self.config.set("detection.heart_rate.update_interval", self.heart_rate_update_interval_spinbox.value())
            
            # Update configuration
            self.config.set_user_settings(self.user_settings)
            
            # Update detection settings in config
            self.config.set("detection.posture.model_complexity", self.posture_model_combo.currentIndex())
            self.config.set("detection.drinking.model_complexity", self.drinking_model_combo.currentIndex())
            
            # Save configuration
            self.config.save()
            
            logger.info("Settings saved successfully")
            
            # Emit signal
            self.settings_saved.emit()
            
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
    
    @pyqtSlot()
    def test_notification(self) -> None:
        """Test notification system."""
        try:
            notification_manager = get_notification_manager()
            notification_manager.send_notification(
                "Test Notification",
                "This is a test notification from the Employee Health Monitoring System.",
                "normal"
            )
            
            logger.info("Test notification sent")
            
        except Exception as e:
            logger.error(f"Error sending test notification: {str(e)}")
    
    @pyqtSlot()
    def test_camera(self) -> None:
        """Test camera access."""
        try:
            # This would normally open a camera preview window
            # For now, just log the action
            camera_id = self.camera_combo.currentIndex()
            logger.info(f"Testing camera with ID: {camera_id}")
            
            # In a real implementation, this would open a dialog with camera preview
            # For example:
            # from ..vision.capture import VideoCapture
            # preview_dialog = CameraPreviewDialog(camera_id, self)
            # preview_dialog.exec()
            
        except Exception as e:
            logger.error(f"Error testing camera: {str(e)}")


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    settings_widget = SettingsWidget()
    settings_widget.setWindowTitle("Settings")
    settings_widget.resize(600, 500)
    settings_widget.show()
    
    sys.exit(app.exec())
