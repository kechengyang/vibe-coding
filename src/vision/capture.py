"""
Video Capture Module for Employee Health Monitoring System.

This module handles webcam access and frame processing.
"""
import cv2
import time
import logging
import threading
from typing import Callable, Optional, Tuple, List

logger = logging.getLogger("employee_health_monitor.vision.capture")

class VideoCapture:
    """Class for capturing video from webcam and processing frames."""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """Initialize the video capture.
        
        Args:
            camera_id: ID of the camera to use (default: 0 for primary webcam)
            resolution: Desired resolution as (width, height)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.is_running = False
        self.processing_thread = None
        self.frame_processors = []
        self.last_frame = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        
        logger.info(f"VideoCapture initialized with camera_id={camera_id}, resolution={resolution}")
    
    def start(self) -> bool:
        """Start video capture.
        
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Video capture already running")
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Start processing thread
            self.is_running = True
            self.start_time = time.time()
            self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.processing_thread.start()
            
            logger.info("Video capture started")
            return True
            
        except Exception as e:
            logger.exception(f"Error starting video capture: {str(e)}")
            return False
    
    def stop(self) -> None:
        """Stop video capture."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Video capture stopped")
    
    def add_frame_processor(self, processor: Callable) -> None:
        """Add a frame processor function.
        
        Args:
            processor: Function that takes a frame and returns a processed frame
        """
        self.frame_processors.append(processor)
        logger.info(f"Added frame processor: {processor.__name__}")
    
    def get_last_frame(self):
        """Get the most recent captured frame.
        
        Returns:
            The last captured frame, or None if no frame has been captured
        """
        return self.last_frame
    
    def get_fps(self) -> float:
        """Get the current frames per second rate.
        
        Returns:
            float: Current FPS
        """
        return self.fps
    
    def _process_frames(self) -> None:
        """Process frames from the video capture (runs in a separate thread)."""
        frame_time = time.time()
        
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Apply all frame processors
                processed_frame = frame.copy()
                for processor in self.frame_processors:
                    try:
                        processed_frame = processor(processed_frame)
                    except Exception as e:
                        logger.exception(f"Error in frame processor {processor.__name__}: {str(e)}")
                
                # Update last frame
                self.last_frame = processed_frame
                
                # Update frame count and FPS
                self.frame_count += 1
                now = time.time()
                if now - frame_time >= 1.0:
                    self.fps = self.frame_count / (now - frame_time)
                    self.frame_count = 0
                    frame_time = now
                
                # Limit processing rate to avoid excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.exception(f"Error processing frame: {str(e)}")
                time.sleep(0.1)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Define a simple frame processor
    def example_processor(frame):
        # Add a timestamp to the frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        return frame
    
    # Create and start video capture
    with VideoCapture() as cap:
        cap.add_frame_processor(example_processor)
        
        # Display frames
        while True:
            frame = cap.get_last_frame()
            if frame is not None:
                cv2.imshow("Video Capture Example", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
