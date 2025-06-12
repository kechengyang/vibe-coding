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
        history_size: int = 15,  # Reduced for faster response
        hand_to_face_threshold: float = 0.15,
        drinking_confidence_threshold: float = 0.55,  # Lowered for better sensitivity
        min_drinking_frames: int = 5,  # Reduced for faster detection
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
        self.hand_position_history = []  # For trajectory analysis
        self.score_history = []          # For trend analysis
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
        
        # Add frame dimensions as text for debugging
        height, width, _ = frame.shape
        cv2.putText(
            annotated_frame,
            f"Frame: {width}x{height}",
            (width - 150, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
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
        
        # Add detailed score information to frame
        y_offset = 30
        cv2.putText(
            annotated_frame,
            f"Drinking Score: {avg_score:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add debug information if available
        if hasattr(self, 'debug_info') and self.debug_info:
            y_offset += 25
            cv2.putText(
                annotated_frame,
                f"Wrist-Mouth Dist: {self.debug_info.get('wrist_to_mouth_dist', 0):.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )
            
            y_offset += 20
            cv2.putText(
                annotated_frame,
                f"Proximity: {self.debug_info.get('proximity_score', 0):.2f} | " +
                f"Hand: {self.debug_info.get('hand_config_score', 0):.2f} | " +
                f"Move: {self.debug_info.get('movement_score', 0):.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
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
        """Calculate the likelihood of a drinking action based on Euclidean distance between wrist and mouth/nose.
        
        Args:
            hand_results: MediaPipe hand detection results
            face_results: MediaPipe face mesh detection results
            
        Returns:
            float: Score between 0 (not drinking) and 1 (likely drinking)
        """
        # If no hands or face detected, return 0
        if not hand_results.multi_hand_landmarks or not face_results.multi_face_landmarks:
            return 0.0
        
        # Get key face landmarks for drinking detection
        # In MediaPipe Face Mesh:
        # - Nose tip is landmark 4
        # - Mouth center can be approximated using the midpoint of upper and lower lip
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Get nose tip coordinates
        nose_tip = face_landmarks.landmark[4]  # Nose tip
        
        # Get mouth center coordinates (approximated)
        # Upper lip top landmark index is 13, lower lip bottom is 14
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_center_x = (upper_lip.x + lower_lip.x) / 2
        mouth_center_y = (upper_lip.y + lower_lip.y) / 2
        mouth_center_z = (upper_lip.z + lower_lip.z) / 2
        
        # Normalize face size to account for different distances from camera
        # Calculate face bounding box
        face_x_coords = [landmark.x for landmark in face_landmarks.landmark]
        face_y_coords = [landmark.y for landmark in face_landmarks.landmark]
        face_width = max(face_x_coords) - min(face_x_coords)
        face_height = max(face_y_coords) - min(face_y_coords)
        face_size = (face_width + face_height) / 2  # Average dimension as size metric
        
        # Initialize minimum distances
        min_wrist_to_mouth_distance = float('inf')
        min_wrist_to_nose_distance = float('inf')
        
        # Calculate hand configuration score
        hand_config_score = 0.0
        
        # Track hand movement
        hand_positions = []
        vertical_movement_score = 0.0
        
        # Process each detected hand
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get wrist position
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Store hand position for trajectory analysis
            hand_positions.append((wrist.x, wrist.y))
            
            # Calculate Euclidean distance from wrist to mouth and nose
            # Normalize by face size to make it scale-invariant
            wrist_to_mouth_distance = (
                ((wrist.x - mouth_center_x) ** 2 + 
                 (wrist.y - mouth_center_y) ** 2 + 
                 (wrist.z - mouth_center_z) ** 2) ** 0.5
            ) / face_size
            
            wrist_to_nose_distance = (
                ((wrist.x - nose_tip.x) ** 2 + 
                 (wrist.y - nose_tip.y) ** 2 + 
                 (wrist.z - nose_tip.z) ** 2) ** 0.5
            ) / face_size
            
            # Update minimum distances
            min_wrist_to_mouth_distance = min(min_wrist_to_mouth_distance, wrist_to_mouth_distance)
            min_wrist_to_nose_distance = min(min_wrist_to_nose_distance, wrist_to_nose_distance)
            
            # Calculate hand configuration (cup-like shape)
            # Get fingertip and middle finger MCP (knuckle) landmarks
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            
            middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            # Calculate average fingertip-to-MCP distance
            avg_fingertip_distance = (
                ((thumb_tip.x - middle_mcp.x) ** 2 + (thumb_tip.y - middle_mcp.y) ** 2) ** 0.5 +
                ((index_tip.x - middle_mcp.x) ** 2 + (index_tip.y - middle_mcp.y) ** 2) ** 0.5 +
                ((middle_tip.x - middle_mcp.x) ** 2 + (middle_tip.y - middle_mcp.y) ** 2) ** 0.5 +
                ((ring_tip.x - middle_mcp.x) ** 2 + (ring_tip.y - middle_mcp.y) ** 2) ** 0.5 +
                ((pinky_tip.x - middle_mcp.x) ** 2 + (pinky_tip.y - middle_mcp.y) ** 2) ** 0.5
            ) / 5
            
            # Calculate finger curl score (lower distance = more curled = more cup-like)
            # Typical values: 0.05-0.1 for curled fingers, 0.15-0.25 for extended fingers
            max_curl_value = 0.25
            min_curl_value = 0.05
            hand_config_score = max(0.0, min(1.0, 1.0 - (
                (avg_fingertip_distance - min_curl_value) / (max_curl_value - min_curl_value)
            )))
        
        # Calculate hand movement (trajectory) score
        if hasattr(self, 'hand_position_history') and self.hand_position_history and hand_positions:
            prev_positions = self.hand_position_history[-min(5, len(self.hand_position_history)):]
            
            if len(prev_positions) > 0 and len(hand_positions) > 0:
                # Calculate vertical movement (positive if moving upward in image coordinates)
                current_y = hand_positions[0][1]
                prev_y = prev_positions[0][1]
                y_movement = prev_y - current_y
                
                # Only consider upward movement (toward face)
                if y_movement > 0:
                    # Normalize movement score
                    vertical_movement_score = min(1.0, y_movement * 10.0)  # Scale factor
        
        # Update hand position history
        self.hand_position_history = getattr(self, 'hand_position_history', [])
        if hand_positions:
            self.hand_position_history.append(hand_positions[0])
            if len(self.hand_position_history) > 30:
                self.hand_position_history.pop(0)
        
        # Convert distance to proximity score (closer = higher score)
        # Threshold values determined through experimentation
        # For mouth distance: ~0.1-0.15 is typical for drinking, ~0.3-0.5 is normal hand position
        mouth_distance_threshold = 0.3
        nose_distance_threshold = 0.35
        
        mouth_proximity_score = max(0.0, 1.0 - (min_wrist_to_mouth_distance / mouth_distance_threshold))
        nose_proximity_score = max(0.0, 1.0 - (min_wrist_to_nose_distance / nose_distance_threshold))
        
        # Take the maximum proximity score (either mouth or nose)
        proximity_score = max(mouth_proximity_score, nose_proximity_score)
        
        # Combine scores with appropriate weights
        # This weighting scheme prioritizes hand-to-mouth proximity but requires
        # supporting evidence from hand configuration and movement
        final_score = (
            0.6 * proximity_score +      # Hand close to mouth/nose
            0.25 * hand_config_score +   # Cup-like hand shape
            0.15 * vertical_movement_score  # Upward hand movement
        )
        
        # Add debug annotations (can be removed in production)
        self.debug_info = {
            'wrist_to_mouth_dist': min_wrist_to_mouth_distance,
            'wrist_to_nose_dist': min_wrist_to_nose_distance,
            'proximity_score': proximity_score,
            'hand_config_score': hand_config_score,
            'movement_score': vertical_movement_score,
            'final_score': final_score
        }
        
        return final_score
    
    def _update_state(self, score: float) -> Optional[DrinkingEvent]:
        """Update the drinking state based on the hand-to-face score with simplified state machine.
        
        Args:
            score: Drinking detection score
            
        Returns:
            DrinkingEvent if a drinking event is detected, None otherwise
        """
        now = time.time()
        event = None
        
        # Store consecutive frames above threshold
        self.consecutive_frames = getattr(self, 'consecutive_frames', 0)
        
        # Store score in history for trend analysis
        self.score_history = getattr(self, 'score_history', [])
        self.score_history.append(score)
        if len(self.score_history) > 30:  # Keep last second of scores (30 frames)
            self.score_history.pop(0)
        
        # Calculate average score for more stability (use shorter window for faster response)
        avg_score = sum(self.score_history[-min(5, len(self.score_history)):]) / min(5, len(self.score_history))
        
        # Simplified state machine that focuses on consecutive frames above threshold
        if self.current_state == DrinkingState.NOT_DRINKING:
            if avg_score >= self.drinking_confidence_threshold:
                # Count consecutive frames above threshold
                self.consecutive_frames += 1
                
                # If we have enough consecutive frames, transition to potential drinking
                if self.consecutive_frames >= 3:  # Need at least 3 frames to start considering
                    self.current_state = DrinkingState.POTENTIAL_DRINKING
                    self.state_start_time = now
                    self.drinking_frames_count = self.consecutive_frames
                    logger.debug("Potential drinking detected")
            else:
                # Reset consecutive frames counter
                self.consecutive_frames = 0
        
        elif self.current_state == DrinkingState.POTENTIAL_DRINKING:
            if avg_score >= self.drinking_confidence_threshold - 0.05:  # Allow slight dip
                # Still in potential drinking state, increment counters
                self.consecutive_frames += 1
                self.drinking_frames_count += 1
                
                # If we've maintained this for enough frames, confirm drinking
                if self.drinking_frames_count >= self.min_drinking_frames:
                    self.current_state = DrinkingState.DRINKING
                    logger.info("Drinking action confirmed")
            else:
                # Score dropped too much, revert to not drinking
                self.current_state = DrinkingState.NOT_DRINKING
                self.consecutive_frames = 0
                self.drinking_frames_count = 0
        
        elif self.current_state == DrinkingState.DRINKING:
            if avg_score < self.drinking_confidence_threshold - 0.1:
                # Hand moved away from face, end drinking event
                duration = now - self.state_start_time
                
                # Only create event if duration is reasonable
                if 0.5 <= duration <= 8.0:  # More restrictive duration check
                    event = DrinkingEvent(
                        timestamp=self.state_start_time,
                        confidence=avg_score,
                        duration=duration
                    )
                    
                    # Record event
                    self.events.append(event)
                    
                    logger.info(f"Drinking event ended, duration: {duration:.2f}s")
                
                # Reset to not drinking
                self.current_state = DrinkingState.NOT_DRINKING
                self.consecutive_frames = 0
                self.drinking_frames_count = 0
            else:
                # Still drinking, update for how long
                self.drinking_frames_count += 1
                
                # If drinking for too long, force end (likely stuck)
                if now - self.state_start_time > 8.0:
                    self.current_state = DrinkingState.NOT_DRINKING
                    self.consecutive_frames = 0
                    logger.info("Drinking event ended (timeout)")
        
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
