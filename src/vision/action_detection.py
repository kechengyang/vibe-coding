"""
Action Detection Module for Employee Health Monitoring System.

This module handles detection of drinking water actions.
"""
import cv2
import time
import logging
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("employee_health_monitor.vision.action_detection")

class DrinkingState(Enum):
    """Enum representing different drinking states."""
    NOT_DRINKING = 0
    POTENTIAL_DRINKING = 1
    DRINKING = 2


@dataclass
class DrinkingEvent:
    """Class for storing drinking event data."""
    timestamp: float
    confidence: float
    duration: float = 0.0


class DrinkingDetector:
    """Class for detecting drinking water actions."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_size: int = 15,
        hand_to_face_threshold: float = 0.15,
        drinking_confidence_threshold: float = 0.7,
        min_drinking_frames: int = 10,
    ):
        """Initialize the drinking detector.
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            history_size: Number of frames to keep in history for smoothing
            hand_to_face_threshold: Threshold for hand-to-face proximity (0-1)
            drinking_confidence_threshold: Threshold for drinking detection (0-1)
            min_drinking_frames: Minimum number of frames to consider as drinking
        """
        # Initialize MediaPipe Hands and Face Mesh
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize detectors
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=2,
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_faces=1,
        )
        
        # Configuration parameters
        self.history_size = history_size
        self.hand_to_face_threshold = hand_to_face_threshold
        self.drinking_confidence_threshold = drinking_confidence_threshold
        self.min_drinking_frames = min_drinking_frames
        
        # State variables
        self.hand_to_face_history = []
        self.current_state = DrinkingState.NOT_DRINKING
        self.state_start_time = time.time()
        self.drinking_frames_count = 0
        self.events = []
        
        logger.info(
            f"DrinkingDetector initialized with hand_to_face_threshold={hand_to_face_threshold}, "
            f"drinking_confidence_threshold={drinking_confidence_threshold}"
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[DrinkingEvent]]:
        """Process a video frame to detect drinking actions.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated frame, drinking event if detected)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hands and face
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        
        # Calculate hand-to-face proximity
        hand_to_face_score = self._calculate_hand_to_face_score(hand_results, face_results)
        
        # Add to history
        self.hand_to_face_history.append(hand_to_face_score)
        if len(self.hand_to_face_history) > self.history_size:
            self.hand_to_face_history.pop(0)
        
        # Calculate average score
        avg_score = sum(self.hand_to_face_history) / len(self.hand_to_face_history) if self.hand_to_face_history else 0
        
        # Add score text to frame
        cv2.putText(
            annotated_frame,
            f"Hand-to-Face: {avg_score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Determine state and check for drinking events
        event = self._update_state(avg_score)
        
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
    
    def _calculate_hand_to_face_score(self, hand_results, face_results) -> float:
        """Calculate a score representing hand-to-face proximity.
        
        Args:
            hand_results: MediaPipe hand detection results
            face_results: MediaPipe face mesh detection results
            
        Returns:
            float: Score between 0 (far) and 1 (close)
        """
        # If no hands or face detected, return 0
        if not hand_results.multi_hand_landmarks or not face_results.multi_face_landmarks:
            return 0.0
        
        # Get face landmarks
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Get face bounding box
        face_x_min = min(landmark.x for landmark in face_landmarks.landmark)
        face_x_max = max(landmark.x for landmark in face_landmarks.landmark)
        face_y_min = min(landmark.y for landmark in face_landmarks.landmark)
        face_y_max = max(landmark.y for landmark in face_landmarks.landmark)
        
        # Expand face bounding box slightly to include area around mouth
        face_width = face_x_max - face_x_min
        face_height = face_y_max - face_y_min
        
        mouth_region_x_min = face_x_min - face_width * 0.1
        mouth_region_x_max = face_x_max + face_width * 0.1
        mouth_region_y_min = face_y_min + face_height * 0.6  # Lower half of face
        mouth_region_y_max = face_y_max + face_height * 0.1
        
        # Check each hand's proximity to mouth region
        max_proximity_score = 0.0
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Use index finger tip and thumb tip as reference points
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            
            # Check if either finger tip is near the mouth region
            for point in [index_tip, thumb_tip]:
                # Check if point is within expanded mouth region
                if (mouth_region_x_min <= point.x <= mouth_region_x_max and
                    mouth_region_y_min <= point.y <= mouth_region_y_max):
                    
                    # Calculate how close the point is to the center of the mouth region
                    mouth_center_x = (mouth_region_x_min + mouth_region_x_max) / 2
                    mouth_center_y = (mouth_region_y_min + mouth_region_y_max) / 2
                    
                    # Calculate distance from point to mouth center (normalized)
                    distance = ((point.x - mouth_center_x) ** 2 + 
                               (point.y - mouth_center_y) ** 2) ** 0.5
                    
                    # Convert distance to proximity score (closer = higher score)
                    max_distance = ((mouth_region_x_max - mouth_region_x_min) ** 2 + 
                                   (mouth_region_y_max - mouth_region_y_min) ** 2) ** 0.5 / 2
                    
                    proximity_score = 1.0 - min(1.0, distance / max_distance)
                    max_proximity_score = max(max_proximity_score, proximity_score)
        
        return max_proximity_score
    
    def _update_state(self, score: float) -> Optional[DrinkingEvent]:
        """Update the drinking state based on the hand-to-face score.
        
        Args:
            score: Hand-to-face proximity score
            
        Returns:
            DrinkingEvent if a drinking event is detected, None otherwise
        """
        now = time.time()
        event = None
        
        if score >= self.drinking_confidence_threshold:
            # Hand is very close to face/mouth
            if self.current_state == DrinkingState.NOT_DRINKING:
                # Transition to potential drinking
                self.current_state = DrinkingState.POTENTIAL_DRINKING
                self.state_start_time = now
                self.drinking_frames_count = 1
                logger.debug("Potential drinking detected")
            
            elif self.current_state == DrinkingState.POTENTIAL_DRINKING:
                # Increment frame count
                self.drinking_frames_count += 1
                
                # Check if we've seen enough frames to confirm drinking
                if self.drinking_frames_count >= self.min_drinking_frames:
                    # Transition to drinking
                    self.current_state = DrinkingState.DRINKING
                    logger.info("Drinking action confirmed")
            
            # Reset timer if already in drinking state
            elif self.current_state == DrinkingState.DRINKING:
                self.state_start_time = now
        
        else:
            # Hand is not close to face/mouth
            if self.current_state == DrinkingState.DRINKING:
                # Create drinking event
                duration = now - self.state_start_time
                event = DrinkingEvent(
                    timestamp=self.state_start_time,
                    confidence=self.drinking_confidence_threshold,
                    duration=duration
                )
                
                # Record event
                self.events.append(event)
                
                logger.info(f"Drinking event ended, duration: {duration:.2f}s")
            
            # Reset to not drinking
            if self.current_state != DrinkingState.NOT_DRINKING:
                self.current_state = DrinkingState.NOT_DRINKING
                self.state_start_time = now
                self.drinking_frames_count = 0
        
        return event
    
    def get_current_state(self) -> DrinkingState:
        """Get the current drinking state.
        
        Returns:
            DrinkingState: Current drinking state
        """
        return self.current_state
    
    def get_state_duration(self) -> float:
        """Get the duration of the current state in seconds.
        
        Returns:
            float: Duration in seconds
        """
        return time.time() - self.state_start_time
    
    def get_events(self) -> List[DrinkingEvent]:
        """Get all recorded drinking events.
        
        Returns:
            List[DrinkingEvent]: List of drinking events
        """
        return self.events.copy()
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.hand_to_face_history = []
        self.current_state = DrinkingState.NOT_DRINKING
        self.state_start_time = time.time()
        self.drinking_frames_count = 0
        self.events = []
        
        logger.info("DrinkingDetector reset")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create video capture
    cap = cv2.VideoCapture(0)
    
    # Create drinking detector
    detector = DrinkingDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, event = detector.process_frame(frame)
        
        # Display frame
        cv2.imshow("Drinking Detection", annotated_frame)
        
        # Print event if drinking detected
        if event:
            print(f"Drinking event detected with confidence {event.confidence:.2f}, duration: {event.duration:.2f}s")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
