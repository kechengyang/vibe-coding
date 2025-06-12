#!/usr/bin/env python
"""
Test script specifically for debugging posture/standing detection issues.

This script creates an instance of PostureDetector and displays detailed
values used in the standing detection algorithm to help diagnose accuracy issues.
"""
import sys
import os
import cv2
import time
import logging
import argparse
import ssl
from pathlib import Path

# Fix for SSL certificate verification error on macOS
def fix_ssl_certificates():
    ssl._create_default_https_context = ssl._create_unverified_context

# Apply SSL fix before importing MediaPipe
fix_ssl_certificates()

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.pose_detection import PostureDetector, PostureState

# Add debug methods to PostureDetector to expose internal values
class DebugPostureDetector(PostureDetector):
    """Extended PostureDetector with debug capabilities."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.debug_info = {}
    
    def process_frame(self, frame):
        """Override process_frame to capture debug information."""
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
            
            # Calculate posture score with debug info
            posture_score, debug_info = self._calculate_posture_score_with_debug(results.pose_landmarks)
            self.debug_info = debug_info
            
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
            
            # Add debug information to frame
            y_offset = 100
            for key, value in debug_info.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"
                
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )
                y_offset += 20
            
            # Check for state change
            event = None
            if new_state != self.current_state:
                # Create event for state change
                event = self.create_event(new_state, avg_score)
                
                # Update state
                self.current_state = new_state
                self.state_start_time = time.time()
            
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
    
    def _calculate_posture_score_with_debug(self, landmarks):
        """Calculate posture score and return detailed debug information."""
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
        
        # Debug info dictionary
        debug_info = {}
        
        # Calculate multiple features for more robust detection
        
        # 1. Torso height ratio (primary feature)
        hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        torso_height = hip_y - shoulder_y
        debug_info['torso_height'] = torso_height
        
        # Adaptive normalization based on visible body parts
        # This helps with different camera angles and distances
        body_height = 0
        visible_parts = 0
        
        # Check if lower body is visible
        lower_body_visible = (
            left_knee.visibility > 0.5 and right_knee.visibility > 0.5 and
            left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5
        )
        debug_info['lower_body_visible'] = lower_body_visible
        
        if lower_body_visible:
            # If full body is visible, use ankle to head distance
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            body_height = ankle_y - shoulder_y
            visible_parts = 3  # shoulders, hips, ankles
            debug_info['ankle_y'] = ankle_y
        else:
            # If only upper body is visible, use hip to shoulder distance as reference
            body_height = hip_y - shoulder_y
            visible_parts = 2  # shoulders, hips
        
        debug_info['body_height'] = body_height
        debug_info['visible_parts'] = visible_parts
        
        # 2. Calculate knee angle if knees are visible (supplementary feature)
        knee_angle_score = 0
        if left_knee.visibility > 0.5 and right_knee.visibility > 0.5:
            # Calculate angle at knee (straight leg = standing, bent knee = sitting)
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            
            # Average knee angle (180 degrees = straight leg, ~90 degrees = sitting)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            debug_info['left_knee_angle'] = left_knee_angle
            debug_info['right_knee_angle'] = right_knee_angle
            debug_info['avg_knee_angle'] = avg_knee_angle
            
            # Normalize: 90 degrees (sitting) -> 0, 180 degrees (standing) -> 1
            knee_angle_score = (avg_knee_angle - 90) / 90
            knee_angle_score = max(0, min(1, knee_angle_score))
            debug_info['knee_angle_score'] = knee_angle_score
        
        # 3. Calculate hip angle (supplementary feature)
        # In standing, the hip angle is straighter
        left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        debug_info['left_hip_angle'] = left_hip_angle
        debug_info['right_hip_angle'] = right_hip_angle
        debug_info['avg_hip_angle'] = avg_hip_angle
        
        # Normalize: 90 degrees (sitting) -> 0, 180 degrees (standing) -> 1
        hip_angle_score = (avg_hip_angle - 90) / 90
        hip_angle_score = max(0, min(1, hip_angle_score))
        debug_info['hip_angle_score'] = hip_angle_score
        
        # 4. Calculate torso vertical alignment (supplementary feature)
        # Standing posture typically has more vertical torso
        left_shoulder_hip_x_diff = abs(left_shoulder.x - left_hip.x)
        right_shoulder_hip_x_diff = abs(right_shoulder.x - right_hip.x)
        avg_x_diff = (left_shoulder_hip_x_diff + right_shoulder_hip_x_diff) / 2
        
        debug_info['left_shoulder_hip_x_diff'] = left_shoulder_hip_x_diff
        debug_info['right_shoulder_hip_x_diff'] = right_shoulder_hip_x_diff
        debug_info['avg_x_diff'] = avg_x_diff
        
        # Normalize: larger x difference (leaning) -> 0, smaller x difference (vertical) -> 1
        alignment_score = 1 - min(1, avg_x_diff * 10)  # Scale factor of 10 for sensitivity
        debug_info['alignment_score'] = alignment_score
        
        # Combine all features with appropriate weights
        # Weights depend on which features are available and reliable
        if lower_body_visible:
            # If lower body is visible, use all features
            # Calculate torso height ratio (correctly normalized)
            torso_height_ratio = min(1.0, torso_height / max(0.01, body_height / visible_parts))
            debug_info['torso_height_ratio'] = torso_height_ratio
            
            final_score = (
                0.3 * torso_height_ratio +  # Torso height ratio
                0.4 * knee_angle_score +    # Knee angle
                0.2 * hip_angle_score +     # Hip angle
                0.1 * alignment_score       # Torso alignment
            )
            
            debug_info['weight_torso'] = 0.3
            debug_info['weight_knee'] = 0.4
            debug_info['weight_hip'] = 0.2
            debug_info['weight_alignment'] = 0.1
        else:
            # If only upper body is visible, rely more on hip angle and alignment
            # For upper body only, use a more conservative approach for torso_height_ratio
            # Since we can't see the legs, we need to rely more on upper body cues
            torso_height_ratio = min(0.7, torso_height / max(0.01, body_height / visible_parts))
            debug_info['torso_height_ratio'] = torso_height_ratio
            
            final_score = (
                0.4 * torso_height_ratio +  # Limited torso height influence
                0.4 * hip_angle_score +     # More weight on hip angle
                0.2 * alignment_score       # More weight on alignment
            )
            
            debug_info['weight_torso'] = 0.4
            debug_info['weight_hip'] = 0.4
            debug_info['weight_alignment'] = 0.2
        
        # Ensure score is in 0-1 range
        final_score = max(0, min(1, final_score))
        debug_info['final_score'] = final_score
        
        # Add threshold values for reference
        debug_info['standing_threshold'] = self.standing_threshold
        debug_info['sitting_threshold'] = self.sitting_threshold
        
        return final_score, debug_info
    
    def create_event(self, new_state, avg_score):
        """Helper method to create a posture event."""
        from src.vision.pose_detection import PostureEvent
        return PostureEvent(
            state=new_state,
            timestamp=time.time(),
            confidence=avg_score if new_state == PostureState.STANDING else 1 - avg_score
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test and debug standing detection")
    parser.add_argument(
        "--model", 
        type=int, 
        choices=[0, 1, 2], 
        default=2,
        help="Posture detection model complexity (0=Lite, 1=Full, 2=Heavy)"
    )
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="Camera ID to use"
    )
    parser.add_argument(
        "--standing-threshold",
        type=float,
        default=None,
        help="Override standing threshold value (0-1)"
    )
    parser.add_argument(
        "--sitting-threshold",
        type=float,
        default=None,
        help="Override sitting threshold value (0-1)"
    )
    parser.add_argument(
        "--debug-level",
        type=str,
        choices=["INFO", "DEBUG"],
        default="DEBUG",
        help="Set logging level"
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.debug_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("test_standing_detection")
    
    # Create video capture
    logger.info(f"Opening camera {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return 1
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera resolution: {width}x{height}, FPS: {fps}")
    
    # Create detector kwargs
    detector_kwargs = {"model_complexity": args.model}
    
    # Add override thresholds if provided
    if args.standing_threshold is not None:
        detector_kwargs["standing_threshold"] = args.standing_threshold
        logger.info(f"Overriding standing threshold: {args.standing_threshold}")
    
    if args.sitting_threshold is not None:
        detector_kwargs["sitting_threshold"] = args.sitting_threshold
        logger.info(f"Overriding sitting threshold: {args.sitting_threshold}")
    
    # Create detector
    logger.info(f"Creating PostureDetector with model_complexity={args.model}")
    detector = DebugPostureDetector(**detector_kwargs)
    
    # Create window
    cv2.namedWindow("Posture Detection Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture Detection Debug", 800, 600)
    
    # Process frames
    logger.info("Starting detection loop. Press 'q' to exit, 's' to take a screenshot")
    frame_count = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break
        
        # Process frame with posture detector
        start_time = time.time()
        processed_frame, event = detector.process_frame(frame)
        processing_time = time.time() - start_time
        
        # Add model info and processing time
        cv2.putText(
            processed_frame,
            f"Model: {'Heavy' if args.model == 2 else 'Full' if args.model == 1 else 'Lite'} ({processing_time:.3f}s)",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        
        # Display frame
        cv2.imshow("Posture Detection Debug", processed_frame)
        
        # Log posture information every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            # Get current posture score
            avg_score = sum(detector.posture_history) / len(detector.posture_history) if detector.posture_history else 0
            
            # Log detailed information
            logger.info(f"Posture Values:")
            logger.info(f"  - Current state: {detector.current_state.name}")
            logger.info(f"  - Average score: {avg_score:.3f}")
            logger.info(f"  - Standing threshold: {detector.standing_threshold:.3f}")
            logger.info(f"  - Sitting threshold: {detector.sitting_threshold:.3f}")
            
            # Log debug info
            if hasattr(detector, 'debug_info') and detector.debug_info:
                for key, value in detector.debug_info.items():
                    if isinstance(value, float):
                        logger.info(f"  - {key}: {value:.3f}")
                    else:
                        logger.info(f"  - {key}: {value}")
        
        # Log events
        if event:
            logger.info(f"Posture event: {event.state.name}, confidence: {event.confidence:.2f}")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Take a screenshot
            screenshot_path = f"posture_debug_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_path, processed_frame)
            logger.info(f"Screenshot saved to {screenshot_path}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
