"""
Camera View Module for Employee Health Monitoring System.

This module defines the camera view UI component for displaying the camera feed.
"""
import logging
import cv2
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

from ..utils.logger import setup_logger

# Set up logger
logger = setup_logger("employee_health_monitor.ui.camera_view")

class CameraView(QWidget):
    """Widget for displaying the camera feed."""
    
    def __init__(self, parent=None):
        """Initialize the camera view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up UI
        self.init_ui()
        
        logger.info("Camera view initialized")
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #000000;")
        
        # Add placeholder text
        self.image_label.setText("Camera feed will appear here when monitoring starts")
        self.image_label.setStyleSheet("""
            color: #ffffff; 
            background-color: #000000; 
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            qproperty-alignment: AlignCenter;
        """)
        
        main_layout.addWidget(self.image_label)
    
    @pyqtSlot(object)
    def update_frame(self, frame: np.ndarray) -> None:
        """Update the displayed frame.
        
        Args:
            frame: OpenCV frame (BGR format)
        """
        if frame is None:
            return
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add a black bar at the bottom for text
            height, width, channels = rgb_frame.shape
            text_bar_height = 40
            frame_with_bar = np.zeros((height + text_bar_height, width, channels), dtype=np.uint8)
            frame_with_bar[:height, :, :] = rgb_frame
            
            # Add text to the black bar
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                frame_with_bar, 
                "Live Camera Feed - All processing happens locally for privacy", 
                (10, height + 30), 
                font, 
                0.7, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Create QImage from frame
            height_with_bar, width, channels = frame_with_bar.shape
            bytes_per_line = channels * width
            q_image = QImage(frame_with_bar.data, width, height_with_bar, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create QPixmap from QImage
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap to fit label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Set pixmap to label
            self.image_label.setPixmap(pixmap)
            
        except Exception as e:
            logger.error(f"Error updating camera frame: {str(e)}")
    
    def clear_frame(self) -> None:
        """Clear the displayed frame."""
        self.image_label.clear()
        self.image_label.setText("Camera feed will appear here when monitoring starts")
        self.image_label.setStyleSheet("""
            color: #ffffff; 
            background-color: #000000; 
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            qproperty-alignment: AlignCenter;
        """)


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    camera_view = CameraView()
    camera_view.setWindowTitle("Camera View")
    camera_view.resize(800, 600)
    camera_view.show()
    
    sys.exit(app.exec())
