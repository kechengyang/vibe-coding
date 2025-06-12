"""
Pose Detection Module for Employee Health Monitoring System.

This module handles detection of standing and sitting postures.
"""
import cv2
import time
import logging
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("employee_health_monitor.vision.pose_detection")

class PostureState(Enum):
    """Enum representing different posture states."""
    UNKNOWN = 0
    SITTING = 1
    STANDING = 2
    TRANSITIONING = 3


@dataclass
class PostureEvent:
    """Class for storing posture event data."""
    state: PostureState
    timestamp: float
    confidence: float


class PostureDetector:
    """Class for detecting sitting and standing postures."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_size: int = 10,
        standing_threshold: float = 0.7,
        sitting_threshold: float = 0.3,
    ):
        """Initialize the posture detector.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            history_size: Number of frames to keep in history for smoothing
            standing_threshold: Threshold for standing detection (0-1)
            sitting_threshold: Threshold for sitting detection (0-1)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        )
        
        self.history_size = history_size
        self.standing_threshold = standing_threshold
        self.sitting_threshold = sitting_threshold
        
        self.posture_history = []
        self.current_state = PostureState.UNKNOWN
        self.state_start_time = time.time()
        self.events = []
        
        logger.info(
            f"PostureDetector initialized with standing_threshold={standing_threshold}, "
            f"sitting_threshold={sitting_threshold}"
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[PostureEvent]]:
        """Process a video frame to detect posture.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated frame, posture event if state changed)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate posture score
            posture_score = self._calculate_posture_score(results.pose_landmarks)
            
            # Add to history
            self.posture_history.append(posture_score)
            if len(self.posture_history) > self.history_size:
                self.posture_history.pop(0)
            
            # Determine posture state
            avg_score = sum(self.posture_history) / len(self.posture_history)
            
            # Add posture score text to frame
            cv2.putText(
                annotated_frame,
                f"Posture Score: {avg_score:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Determine state
            new_state = self._determine_state(avg_score)
            
            # Check for state change
            event = None
            if new_state != self.current_state:
                # Create event for state change
                event = PostureEvent(
                    state=new_state,
                    timestamp=time.time(),
                    confidence=avg_score if new_state == PostureState.STANDING else 1 - avg_score
                )
                
                # Record event
                self.events.append(event)
                
                # Update state
                self.current_state = new_state
                self.state_start_time = time.time()
                
                logger.info(f"Posture state changed to {new_state.name} with confidence {event.confidence:.2f}")
            
            # Add current state text to frame
            cv2.putText(
                annotated_frame,
                f"State: {self.current_state.name}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            return annotated_frame, event
        
        return annotated_frame, None
    
    def _calculate_posture_score(self, landmarks) -> float:
        """Calculate a score representing the likelihood of standing vs sitting.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Score between 0 (sitting) and 1 (standing)
        """
        # Get relevant landmarks
        # Hip and shoulder landmarks are used to determine posture
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate average y-positions
        hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Calculate distance between hips and shoulders
        # When standing, this distance is larger relative to the frame height
        distance = hip_y - shoulder_y
        
        # Normalize to 0-1 range
        # These thresholds may need adjustment based on camera position and user
        min_distance = 0.1  # Minimum expected distance when sitting
        max_distance = 0.3  # Maximum expected distance when standing
        
        normalized_score = (distance - min_distance) / (max_distance - min_distance)
        normalized_score = max(0, min(1, normalized_score))
        
        return normalized_score
    
    def _determine_state(self, score: float) -> PostureState:
        """Determine posture state based on score.
        
        Args:
            score: Posture score between 0 and 1
            
        Returns:
            PostureState: Determined posture state
        """
        if score >= self.standing_threshold:
            return PostureState.STANDING
        elif score <= self.sitting_threshold:
            return PostureState.SITTING
        else:
            return PostureState.TRANSITIONING
    
    def get_current_state(self) -> PostureState:
        """Get the current posture state.
        
        Returns:
            PostureState: Current posture state
        """
        return self.current_state
    
    def get_state_duration(self) -> float:
        """Get the duration of the current state in seconds.
        
        Returns:
            float: Duration in seconds
        """
        return time.time() - self.state_start_time
    
    def get_events(self) -> List[PostureEvent]:
        """Get all recorded posture events.
        
        Returns:
            List[PostureEvent]: List of posture events
        """
        return self.events.copy()
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.posture_history = []
        self.current_state = PostureState.UNKNOWN
        self.state_start_time = time.time()
        self.events = []
        
        logger.info("PostureDetector reset")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create video capture
    cap = cv2.VideoCapture(0)
    
    # Create posture detector
    detector = PostureDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, event = detector.process_frame(frame)
        
        # Display frame
        cv2.imshow("Posture Detection", annotated_frame)
        
        # Print event if state changed
        if event:
            print(f"Posture changed to {event.state.name} with confidence {event.confidence:.2f}")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
