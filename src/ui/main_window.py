"""
Main Window for Employee Health Monitoring System.

This module defines the main application window and UI components.
"""
import sys
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QStatusBar, QMessageBox,
    QDialog, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QAction, QPixmap, QImage

from .dashboard import Dashboard
from .settings import SettingsWidget
from .camera_view import CameraView
from .heart_rate_view import HeartRateView
from ..vision.capture import VideoCapture
from ..vision.pose_detection import PostureDetector, PostureState
from ..vision.action_detection import DrinkingDetector, DrinkingState
from ..vision.heart_rate_detection import HeartRateDetector, HeartRateEvent
from ..data.database import Database
from ..data.models import PostureEvent, DrinkingEvent, HeartRateEvent
from ..utils.config import get_config
from ..utils.notifications import get_notification_manager
from ..utils.logger import setup_logger

# Set up logger
logger = setup_logger("employee_health_monitor.ui.main_window")

class MonitoringThread(QThread):
    """Thread for running the health monitoring system."""
    
    # Signals
    frame_ready = pyqtSignal(object)
    posture_changed = pyqtSignal(object)
    drinking_detected = pyqtSignal(object)
    heart_rate_updated = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the monitoring thread.
        
        Args:
            parent: Parent object
        """
        super().__init__(parent)
        
        # Get configuration
        self.config = get_config()
        
        # Initialize components
        self.video_capture = None
        self.posture_detector = None
        self.drinking_detector = None
        self.heart_rate_detector = None
        self.db = None
        
        # State variables
        self.is_running = False
        self.current_posture_state = PostureState.UNKNOWN
        self.posture_state_start_time = 0
        self.sitting_duration = 0
        
        logger.info("Monitoring thread initialized")
    
    def run(self):
        """Run the monitoring thread."""
        try:
            # Initialize database
            self.db = Database()
            
            # Initialize video capture
            camera_id = self.config.get("user_settings.camera_id", 0)
            resolution = self.config.get("camera.resolution", [640, 480])
            
            self.video_capture = VideoCapture(camera_id=camera_id, resolution=tuple(resolution))
            
            # Initialize detectors
            self.posture_detector = PostureDetector(
                min_detection_confidence=self.config.get("detection.posture.min_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("detection.posture.min_tracking_confidence", 0.5),
                history_size=self.config.get("detection.posture.history_size", 10),
                standing_threshold=self.config.get("detection.posture.standing_threshold", 0.7),
                sitting_threshold=self.config.get("detection.posture.sitting_threshold", 0.3)
            )
            
            self.drinking_detector = DrinkingDetector(
                min_detection_confidence=self.config.get("detection.drinking.min_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("detection.drinking.min_tracking_confidence", 0.5),
                history_size=self.config.get("detection.drinking.history_size", 15),
                hand_to_face_threshold=self.config.get("detection.drinking.hand_to_face_threshold", 0.15),
                drinking_confidence_threshold=self.config.get("detection.drinking.drinking_confidence_threshold", 0.7),
                min_drinking_frames=self.config.get("detection.drinking.min_drinking_frames", 10)
            )
            
            self.heart_rate_detector = HeartRateDetector(
                buffer_size=self.config.get("detection.heart_rate.buffer_size", 300),
                min_detection_confidence=self.config.get("detection.heart_rate.min_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("detection.heart_rate.min_tracking_confidence", 0.5),
                update_interval=self.config.get("detection.heart_rate.update_interval", 30),
                min_bpm=self.config.get("detection.heart_rate.min_bpm", 45),
                max_bpm=self.config.get("detection.heart_rate.max_bpm", 240)
            )
            
            # Start video capture
            if not self.video_capture.start():
                self.error_occurred.emit("Failed to start video capture")
                return
            
            # Set running flag
            self.is_running = True
            
            # Process frames
            while self.is_running:
                # Get frame
                frame = self.video_capture.get_last_frame()
                
                if frame is not None:
                    # Process frame for posture detection
                    posture_frame, posture_event = self.posture_detector.process_frame(frame)
                    
                    # Process frame for drinking detection
                    drinking_frame, drinking_event = self.drinking_detector.process_frame(posture_frame)
                    
                    # Process frame for heart rate detection
                    heart_rate_frame, heart_rate_event = self.heart_rate_detector.process_frame(drinking_frame)
                    
                    # Emit frame
                    self.frame_ready.emit(heart_rate_frame)
                    
                    # Handle posture event
                    if posture_event:
                        self.handle_posture_event(posture_event)
                    
                    # Handle drinking event
                    if drinking_event:
                        self.handle_drinking_event(drinking_event)
                    
                    # Handle heart rate event
                    if heart_rate_event:
                        self.handle_heart_rate_event(heart_rate_event)
                    
                    # Update sitting duration
                    self.update_sitting_duration()
                    
                    # Check for reminders
                    self.check_reminders()
                
                # Sleep to avoid high CPU usage
                self.msleep(10)
            
            # Clean up
            self.video_capture.stop()
            
            if self.db:
                self.db.close()
            
            logger.info("Monitoring thread stopped")
            
        except Exception as e:
            logger.exception(f"Error in monitoring thread: {str(e)}")
            self.error_occurred.emit(f"Error in monitoring thread: {str(e)}")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.is_running = False
        self.wait()
    
    def handle_posture_event(self, event: PostureEvent):
        """Handle posture event.
        
        Args:
            event: Posture event
        """
        # Update state
        old_state = self.current_posture_state
        self.current_posture_state = event.state
        
        # Record state change time
        now = time.time()
        state_duration = now - self.posture_state_start_time
        self.posture_state_start_time = now
        
        # Log event
        logger.info(f"Posture changed from {old_state.name} to {event.state.name}")
        
        # Add to database if state was STANDING and changed to something else
        if old_state == PostureState.STANDING and event.state != PostureState.STANDING:
            # Add standing event to database
            try:
                self.db.add_standing_event(
                    start_time=self.posture_state_start_time - state_duration,
                    end_time=self.posture_state_start_time,
                    duration=state_duration,
                    confidence=event.confidence
                )
                
                # Send notification if stood for a significant time
                if state_duration >= 60:  # At least 1 minute
                    notification_manager = get_notification_manager()
                    notification_manager.send_standing_achievement(state_duration / 60)
                
                logger.info(f"Added standing event with duration {state_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error adding standing event to database: {str(e)}")
        
        # Reset sitting duration if now standing
        if event.state == PostureState.STANDING:
            self.sitting_duration = 0
        
        # Emit signal
        self.posture_changed.emit(event)
    
    def handle_drinking_event(self, event: DrinkingEvent):
        """Handle drinking event.
        
        Args:
            event: Drinking event
        """
        # Log event
        logger.info(f"Drinking event detected with duration {event.duration:.2f}s")
        
        # Add to database
        try:
            event_id = self.db.add_drinking_event(
                timestamp=event.timestamp,
                duration=event.duration,
                confidence=event.confidence
            )
            
            # Get today's drinking count
            today = datetime.now().date()
            drinking_events = self.db.get_drinking_events(today, today)
            drinking_count = len(drinking_events)
            
            # Send notification
            notification_manager = get_notification_manager()
            notification_manager.send_drinking_achievement(drinking_count)
            
            logger.info(f"Added drinking event with ID {event_id}")
            
        except Exception as e:
            logger.error(f"Error adding drinking event to database: {str(e)}")
        
        # Emit signal
        self.drinking_detected.emit(event)
    
    def handle_heart_rate_event(self, event: HeartRateEvent):
        """Handle heart rate event.
        
        Args:
            event: Heart rate event
        """
        # Log event
        logger.info(f"Heart rate updated: {event.bpm:.1f} BPM (confidence: {event.confidence:.2f})")
        
        # Add to database
        try:
            event_id = self.db.add_heart_rate_event(
                timestamp=event.timestamp,
                bpm=event.bpm,
                confidence=event.confidence
            )
            
            logger.info(f"Added heart rate event with ID {event_id}")
            
        except Exception as e:
            logger.error(f"Error adding heart rate event to database: {str(e)}")
        
        # Emit signal
        self.heart_rate_updated.emit(event)
    
    def update_sitting_duration(self):
        """Update sitting duration."""
        if self.current_posture_state == PostureState.SITTING:
            self.sitting_duration = time.time() - self.posture_state_start_time
    
    def check_reminders(self):
        """Check if reminders should be sent."""
        notification_manager = get_notification_manager()
        notification_manager.check_and_send_reminders(
            self.current_posture_state.name,
            self.sitting_duration
        )


class MainWindow(QApplication):
    """Main application window for the Employee Health Monitoring System."""
    
    def __init__(self, argv):
        """Initialize the application and main window.
        
        Args:
            argv: Command line arguments
        """
        super().__init__(argv)
        
        # Set application info
        self.setApplicationName("Employee Health Monitor")
        self.setApplicationVersion("0.1.0")
        
        # Create the main window
        self.window = MainWindowWidget()
        self.window.show()
        
        # Start monitoring if configured
        config = get_config()
        if config.get("user_settings.start_monitoring_on_launch", False):
            self.window.start_monitoring()
        
        logger.info("Main window initialized")


class MainWindowWidget(QMainWindow):
    """Main window widget for the application."""
    
    def __init__(self):
        """Initialize the main window widget."""
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Employee Health Monitor")
        self.setMinimumSize(1000, 700)
        
        # Initialize components
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Set up the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create the tab widget for different views
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_camera_tab()
        self.create_heart_rate_tab()
        self.create_settings_tab()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create toolbar with controls
        self.create_toolbar()
        
        logger.info("Main window widget created")
    
    def create_dashboard_tab(self):
        """Create the dashboard tab with statistics and visualizations."""
        self.dashboard = Dashboard(self)
        self.tab_widget.addTab(self.dashboard, "Dashboard")
    
    def create_camera_tab(self):
        """Create the camera tab with live video feed."""
        self.camera_view = CameraView(self)
        self.tab_widget.addTab(self.camera_view, "Camera")
    
    def create_heart_rate_tab(self):
        """Create the heart rate tab with heart rate data."""
        self.heart_rate_view = HeartRateView(self)
        self.tab_widget.addTab(self.heart_rate_view, "Heart Rate")
    
    def create_settings_tab(self):
        """Create the settings tab with configuration options."""
        self.settings_widget = SettingsWidget(self)
        self.settings_widget.settings_saved.connect(self.on_settings_saved)
        self.tab_widget.addTab(self.settings_widget, "Settings")
    
    def create_toolbar(self):
        """Create the toolbar with control buttons."""
        toolbar = self.addToolBar("Controls")
        toolbar.setMovable(False)
        
        # Start monitoring action
        self.start_action = QAction("Start Monitoring", self)
        self.start_action.triggered.connect(self.start_monitoring)
        toolbar.addAction(self.start_action)
        
        # Stop monitoring action
        self.stop_action = QAction("Stop Monitoring", self)
        self.stop_action.triggered.connect(self.stop_monitoring)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)
    
    @pyqtSlot()
    def start_monitoring(self):
        """Start the health monitoring system."""
        if self.is_monitoring:
            return
        
        try:
            # Create monitoring thread
            self.monitoring_thread = MonitoringThread(self)
            
            # Connect signals
            self.monitoring_thread.error_occurred.connect(self.on_monitoring_error)
            self.monitoring_thread.frame_ready.connect(self.camera_view.update_frame)
            self.monitoring_thread.heart_rate_updated.connect(self.on_heart_rate_updated)
            
            # Start thread
            self.monitoring_thread.start()
            
            # Update UI
            self.is_monitoring = True
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(True)
            self.status_bar.showMessage("Monitoring started")
            
            logger.info("Monitoring started")
            
            # Show a message box about camera access
            QMessageBox.information(
                self,
                "Monitoring Started",
                "The system will now use your camera to monitor standing and drinking activities.\n\n"
                "All processing happens locally on your device for privacy."
            )
            
        except Exception as e:
            logger.exception(f"Error starting monitoring: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start monitoring: {str(e)}"
            )
    
    @pyqtSlot()
    def stop_monitoring(self):
        """Stop the health monitoring system."""
        if not self.is_monitoring:
            return
        
        try:
            # Stop monitoring thread
            if self.monitoring_thread:
                self.monitoring_thread.stop()
                self.monitoring_thread = None
            
            # Update UI
            self.is_monitoring = False
            self.start_action.setEnabled(True)
            self.stop_action.setEnabled(False)
            self.status_bar.showMessage("Monitoring stopped")
            
            # Clear camera view
            self.camera_view.clear_frame()
            
            logger.info("Monitoring stopped")
            
        except Exception as e:
            logger.exception(f"Error stopping monitoring: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to stop monitoring: {str(e)}"
            )
    
    @pyqtSlot(str)
    def on_monitoring_error(self, error_message: str):
        """Handle monitoring error.
        
        Args:
            error_message: Error message
        """
        # Stop monitoring
        self.stop_monitoring()
        
        # Show error message
        QMessageBox.critical(
            self,
            "Monitoring Error",
            error_message
        )
    
    @pyqtSlot(object)
    def on_heart_rate_updated(self, event: HeartRateEvent):
        """Handle heart rate updated event.
        
        Args:
            event: Heart rate event
        """
        # Update status bar with current heart rate
        self.status_bar.showMessage(f"Heart Rate: {event.bpm:.1f} BPM", 3000)
    
    @pyqtSlot()
    def on_settings_saved(self):
        """Handle settings saved event."""
        # Show confirmation message
        self.status_bar.showMessage("Settings saved", 3000)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Ask for confirmation before closing
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit the application?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Stop monitoring if running
            if self.is_monitoring:
                self.stop_monitoring()
            
            logger.info("Application closing")
            event.accept()
        else:
            event.ignore()
