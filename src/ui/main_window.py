"""
Main Window for Employee Health Monitoring System.

This module defines the main application window and UI components.
"""
import sys
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QStatusBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QIcon, QAction

logger = logging.getLogger("employee_health_monitor.ui.main_window")

class MainWindow(QApplication):
    """Main application window for the Employee Health Monitoring System."""
    
    def __init__(self, argv):
        """Initialize the application and main window.
        
        Args:
            argv: Command line arguments
        """
        super().__init__(argv)
        
        self.setApplicationName("Employee Health Monitor")
        self.setApplicationVersion("0.1.0")
        
        # Create the main window
        self.window = MainWindowWidget()
        self.window.show()
        
        # Set up a timer to periodically update the UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.window.update_stats)
        self.update_timer.start(1000)  # Update every second
        
        logger.info("Main window initialized")

class MainWindowWidget(QMainWindow):
    """Main window widget for the application."""
    
    def __init__(self):
        """Initialize the main window widget."""
        super().__init__()
        
        self.setWindowTitle("Employee Health Monitor")
        self.setMinimumSize(800, 600)
        
        # Set up the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create the tab widget for different views
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_dashboard_tab()
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
        dashboard_widget = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_widget)
        
        # Add title
        title_label = QLabel("Health Monitoring Dashboard")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        dashboard_layout.addWidget(title_label)
        
        # Add placeholder for statistics
        self.stats_label = QLabel("No data collected yet. Monitoring will begin when you start the system.")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 14px; margin: 20px;")
        dashboard_layout.addWidget(self.stats_label)
        
        # Add placeholder for charts
        charts_label = QLabel("Charts will appear here once data is collected.")
        charts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dashboard_layout.addWidget(charts_label)
        
        # Add to tab widget
        self.tab_widget.addTab(dashboard_widget, "Dashboard")
    
    def create_settings_tab(self):
        """Create the settings tab with configuration options."""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # Add title
        title_label = QLabel("Settings")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        settings_layout.addWidget(title_label)
        
        # Add placeholder for settings
        settings_label = QLabel("Configuration options will be available here.")
        settings_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_layout.addWidget(settings_label)
        
        # Add to tab widget
        self.tab_widget.addTab(settings_widget, "Settings")
    
    def create_toolbar(self):
        """Create the toolbar with control buttons."""
        toolbar = self.addToolBar("Controls")
        toolbar.setMovable(False)
        
        # Start monitoring action
        start_action = QAction("Start Monitoring", self)
        start_action.triggered.connect(self.start_monitoring)
        toolbar.addAction(start_action)
        
        # Stop monitoring action
        stop_action = QAction("Stop Monitoring", self)
        stop_action.triggered.connect(self.stop_monitoring)
        toolbar.addAction(stop_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)
    
    @pyqtSlot()
    def start_monitoring(self):
        """Start the health monitoring system."""
        # This would initialize the camera and start monitoring
        # For now, just update the status
        self.status_bar.showMessage("Monitoring started")
        logger.info("Monitoring started")
        
        # Show a message box about camera access
        QMessageBox.information(
            self,
            "Monitoring Started",
            "The system will now use your camera to monitor standing and drinking activities.\n\n"
            "All processing happens locally on your device for privacy."
        )
    
    @pyqtSlot()
    def stop_monitoring(self):
        """Stop the health monitoring system."""
        # This would stop the camera and monitoring
        # For now, just update the status
        self.status_bar.showMessage("Monitoring stopped")
        logger.info("Monitoring stopped")
    
    @pyqtSlot()
    def update_stats(self):
        """Update the statistics display."""
        # This would update with real data from the monitoring system
        # For now, just a placeholder
        pass
    
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
            logger.info("Application closing")
            event.accept()
        else:
            event.ignore()
