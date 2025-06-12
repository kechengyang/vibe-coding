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
from src.utils.config import get_config

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
        # Default values will be overridden by config below
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_size: int = 20,
        standing_threshold: float = 0.65,
        sitting_threshold: float = 0.35,
        model_complexity: int = 1,
    ):
        """Initialize the posture detector.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            history_size: Number of frames to keep in history for smoothing
            standing_threshold: Threshold for standing detection (0-1)
            sitting_threshold: Threshold for sitting detection (0-1)
            model_complexity: MediaPipe model complexity (0=Lite, 1=Full, 2=Heavy)
        """
        config = get_config()

        # Load parameters from config
        _min_detection_confidence = config.get("detection.posture.min_detection_confidence", min_detection_confidence)
        _min_tracking_confidence = config.get("detection.posture.min_tracking_confidence", min_tracking_confidence)
        _history_size = config.get("detection.posture.history_size", history_size)
        _standing_threshold = config.get("detection.posture.standing_threshold", standing_threshold)
        _sitting_threshold = config.get("detection.posture.sitting_threshold", sitting_threshold)
        _model_complexity = config.get("detection.posture.model_complexity", model_complexity)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=_min_detection_confidence,
            min_tracking_confidence=_min_tracking_confidence,
            model_complexity=_model_complexity,  # 0=Lite, 1=Full, 2=Heavy
        )
        
        logger.info(f"Using Pose model with complexity level: {_model_complexity}")
        
        self.history_size = _history_size
        self.standing_threshold = _standing_threshold
        self.sitting_threshold = _sitting_threshold
        
        self.posture_history = []
        self.current_state = PostureState.UNKNOWN
        self.state_start_time = time.time()
        self.events = []
        
        logger.info(
            f"PostureDetector initialized with history_size={self.history_size}, "
            f"standing_threshold={self.standing_threshold}, sitting_threshold={self.sitting_threshold}"
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
            
            # Draw additional information about joint angles
            try:
                # Get landmarks
                left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                
                # Calculate angle
                if left_hip.visibility > 0.5 and left_knee.visibility > 0.5 and left_ankle.visibility > 0.5:
                    angle = self._calculate_angle(left_hip, left_knee, left_ankle)
                    
                    # Draw angle text
                    knee_x = int(left_knee.x * annotated_frame.shape[1])
                    knee_y = int(left_knee.y * annotated_frame.shape[0])
                    cv2.putText(
                        annotated_frame,
                        f"{angle:.0f}°",
                        (knee_x + 10, knee_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1
                    )
            except Exception as e:
                # Ignore errors in drawing additional information
                pass
            
            # Calculate posture score
            posture_score = self._calculate_posture_score(results.pose_landmarks)
            
            # Add to history
            self.posture_history.append(posture_score)
            if len(self.posture_history) > self.history_size:
                self.posture_history.pop(0)
            
            # Determine posture state
            avg_score = sum(self.posture_history) / len(self.posture_history)
            
            # Get frame dimensions
            height, width, _ = annotated_frame.shape
            
            # Add semi-transparent background for better readability
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, height-65), (200, height-5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
            
            # Add section title
            cv2.putText(
                annotated_frame,
                "POSTURE",
                (10, height-50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Add posture score text to frame
            cv2.putText(
                annotated_frame,
                f"Score: {avg_score:.2f}",
                (10, height-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
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
                (10, height-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
            
            return annotated_frame, event
        
        return annotated_frame, None
    
    def _calculate_posture_score(self, landmarks) -> float:
        """Calculate a score representing the likelihood of standing vs sitting.
        This improved version uses a simplified rule-based approach focusing on knee angles
        and hip-to-ankle height ratio.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Score between 0 (sitting) and 1 (standing)
        """
        # Get relevant landmarks for enhanced posture detection
        # Upper body landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Lower body landmarks (if visible)
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate multiple features for more robust detection
        
        # 1. Torso height ratio (primary feature)
        hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        torso_height = hip_y - shoulder_y
        
        # Adaptive normalization based on visible body parts
        # This helps with different camera angles and distances
        body_height = 0
        visible_parts = 0
        
        # Check if lower body is visible
        lower_body_visible = (
            left_knee.visibility > 0.5 and right_knee.visibility > 0.5 and
            left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5
        )
        
        if lower_body_visible:
            # If full body is visible, use ankle to head distance
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            body_height = ankle_y - shoulder_y
            visible_parts = 3  # shoulders, hips, ankles
        else:
            # If only upper body is visible, use hip to shoulder distance as reference
            body_height = hip_y - shoulder_y
            visible_parts = 2  # shoulders, hips
        
        # 2. Calculate knee angle if knees are visible (supplementary feature)
        knee_angle_score = 0
        if left_knee.visibility > 0.5 and right_knee.visibility > 0.5:
            # Calculate angle at knee (straight leg = standing, bent knee = sitting)
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            
            # Average knee angle (180 degrees = straight leg, ~90 degrees = sitting)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            # Normalize: 90 degrees (sitting) -> 0, 180 degrees (standing) -> 1
            knee_angle_score = (avg_knee_angle - 90) / 90
            knee_angle_score = max(0, min(1, knee_angle_score))
        
        # 3. Calculate hip angle (supplementary feature)
        # In standing, the hip angle is straighter
        left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # Normalize: 90 degrees (sitting) -> 0, 180 degrees (standing) -> 1
        hip_angle_score = (avg_hip_angle - 90) / 90
        hip_angle_score = max(0, min(1, hip_angle_score))
        
        # 4. Calculate torso vertical alignment (supplementary feature)
        # Standing posture typically has more vertical torso
        left_shoulder_hip_x_diff = abs(left_shoulder.x - left_hip.x)
        right_shoulder_hip_x_diff = abs(right_shoulder.x - right_hip.x)
        avg_x_diff = (left_shoulder_hip_x_diff + right_shoulder_hip_x_diff) / 2
        
        # Normalize: larger x difference (leaning) -> 0, smaller x difference (vertical) -> 1
        alignment_score = 1 - min(1, avg_x_diff * 10)  # Scale factor of 10 for sensitivity
        
        # Combine all features with appropriate weights
        # Weights depend on which features are available and reliable
        if lower_body_visible:
            # If lower body is visible, use all features
            final_score = (
                0.3 * (torso_height / (body_height / visible_parts)) +  # Torso height ratio
                0.4 * knee_angle_score +                                # Knee angle
                0.2 * hip_angle_score +                                 # Hip angle
                0.1 * alignment_score                                   # Torso alignment
            )
        else:
            # If only upper body is visible, rely more on torso features
            final_score = (
                0.6 * (torso_height / (body_height / visible_parts)) +  # Torso height ratio
                0.3 * hip_angle_score +                                 # Hip angle
                0.1 * alignment_score                                   # Torso alignment
            )
        
        # Ensure score is in 0-1 range
        final_score = max(0, min(1, final_score))
        
        return final_score
    
    def _calculate_angle(self, a, b, c) -> float:
        """Calculate the angle between three points (in degrees).
        
        Args:
            a: First point
            b: Middle point (vertex)
            c: Third point
            
        Returns:
            float: Angle in degrees
        """
        # Convert landmarks to numpy arrays
        a_vec = np.array([a.x, a.y, a.z if hasattr(a, 'z') else 0])
        b_vec = np.array([b.x, b.y, b.z if hasattr(b, 'z') else 0])
        c_vec = np.array([c.x, c.y, c.z if hasattr(c, 'z') else 0])
        
        # Calculate vectors
        ba = a_vec - b_vec
        bc = c_vec - b_vec
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(ba, bc)
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)
        
        # Calculate angle in radians and convert to degrees
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        cos_angle = max(-1, min(1, cos_angle))  # Ensure value is in valid range
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _determine_state(self, score: float) -> PostureState:
        """Determine posture state based on score with hysteresis to prevent oscillation.
        
        Args:
            score: Posture score between 0 and 1
            
        Returns:
            PostureState: Determined posture state
        """
        # Apply hysteresis based on current state to prevent oscillation
        # This creates a "sticky" state that requires more change to transition
        if self.current_state == PostureState.STANDING:
            # When already standing, require a lower score to transition down
            if score <= self.sitting_threshold:
                return PostureState.SITTING
            elif score < self.standing_threshold - 0.1:  # Hysteresis buffer
                return PostureState.TRANSITIONING
            else:
                return PostureState.STANDING
                
        elif self.current_state == PostureState.SITTING:
            # When already sitting, require a higher score to transition up
            if score >= self.standing_threshold:
                return PostureState.STANDING
            elif score > self.sitting_threshold + 0.1:  # Hysteresis buffer
                return PostureState.TRANSITIONING
            else:
                return PostureState.SITTING
                
        elif self.current_state == PostureState.TRANSITIONING:
            # From transitioning, use standard thresholds
            if score >= self.standing_threshold:
                return PostureState.STANDING
            elif score <= self.sitting_threshold:
                return PostureState.SITTING
            else:
                return PostureState.TRANSITIONING
                
        else:  # UNKNOWN or initial state
            # Use standard thresholds for initial determination
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
