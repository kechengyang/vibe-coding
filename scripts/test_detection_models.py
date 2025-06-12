#!/usr/bin/env python
"""
Test script for verifying different MediaPipe model complexities.

This script creates instances of PostureDetector and DrinkingDetector
with different model complexity settings and displays the video feed
with detection overlays.
"""
import sys
import os
import cv2
import time
import logging
import argparse
import ssl
import urllib.request
from pathlib import Path

# Fix for SSL certificate verification error on macOS
# This is needed to download the MediaPipe models
def fix_ssl_certificates():
    # Bypass SSL verification (not recommended for production)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Alternative: Set SSL certificate file path
    # If you have a valid certificate file, you can use this instead
    # os.environ['SSL_CERT_FILE'] = '/path/to/cert.pem'
    
    # Another alternative for macOS Python installed via Homebrew
    # cert_path = Path(sys._MEIPASS).joinpath('lib/python3.9/site-packages/certifi/cacert.pem')
    # os.environ['SSL_CERT_FILE'] = str(cert_path)

# Apply SSL fix before importing MediaPipe
fix_ssl_certificates()

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.pose_detection import PostureDetector
from src.vision.action_detection import DrinkingDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test MediaPipe model complexities")
    parser.add_argument(
        "--posture-model", 
        type=int, 
        choices=[0, 1, 2], 
        default=2,
        help="Posture detection model complexity (0=Lite, 1=Full, 2=Heavy)"
    )
    parser.add_argument(
        "--drinking-model", 
        type=int, 
        choices=[0, 1, 2], 
        default=1,
        help="Drinking detection model complexity (0=Lite, 1=Full, 2=Heavy)"
    )
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="Camera ID to use"
    )
    parser.add_argument(
        "--ssl-verify",
        action="store_true",
        help="Enable SSL certificate verification (default: disabled)"
    )
    return parser.parse_args()

def main():
    """Main function."""
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed output
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("test_detection_models")
    
    # Parse command line arguments
    args = parse_args()
    
    # Re-enable SSL verification if requested
    if args.ssl_verify:
        ssl._create_default_https_context = ssl.create_default_context
        logger.info("SSL certificate verification enabled")
    else:
        logger.info("SSL certificate verification disabled")
    
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
    
    # Create detectors with specified model complexities
    logger.info(f"Creating PostureDetector with model_complexity={args.posture_model}")
    posture_detector = PostureDetector(model_complexity=args.posture_model)
    
    logger.info(f"Creating DrinkingDetector with model_complexity={args.drinking_model}")
    drinking_detector = DrinkingDetector(model_complexity=args.drinking_model)
    
    # Create windows
    cv2.namedWindow("Posture Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Drinking Detection", cv2.WINDOW_NORMAL)
    
    # Resize windows to fit screen
    cv2.resizeWindow("Posture Detection", 640, 480)
    cv2.resizeWindow("Drinking Detection", 640, 480)
    
    # Process frames
    logger.info("Starting detection loop. Press 'q' to exit.")
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break
        
        # Process frame with posture detector
        posture_start = time.time()
        posture_frame, posture_event = posture_detector.process_frame(frame)
        posture_time = time.time() - posture_start
        
        # Process frame with drinking detector
        drinking_start = time.time()
        drinking_frame, drinking_event = drinking_detector.process_frame(frame)
        drinking_time = time.time() - drinking_start
        
        # Print debug info about drinking detection
        if hasattr(drinking_detector, 'debug_info') and drinking_detector.debug_info:
            debug_info = drinking_detector.debug_info
            logger.info(f"Drinking Detection Values:")
            logger.info(f"  - Wrist to mouth distance: {debug_info.get('wrist_to_mouth_dist', 'N/A'):.3f}")
            logger.info(f"  - Wrist to nose distance: {debug_info.get('wrist_to_nose_dist', 'N/A'):.3f}")
            logger.info(f"  - Proximity score: {debug_info.get('proximity_score', 'N/A'):.3f}")
            logger.info(f"  - Hand config score: {debug_info.get('hand_config_score', 'N/A'):.3f}")
            logger.info(f"  - Movement score: {debug_info.get('movement_score', 'N/A'):.3f}")
            logger.info(f"  - Final score: {debug_info.get('final_score', 'N/A'):.3f}")
            
        # Get and print current drinking state and threshold
        current_state = drinking_detector.current_state.name
        threshold = drinking_detector.drinking_confidence_threshold
        avg_score = sum(drinking_detector.hand_to_face_history) / len(drinking_detector.hand_to_face_history) if drinking_detector.hand_to_face_history else 0
        
        logger.info(f"Current state: {current_state}, Avg score: {avg_score:.3f}, Threshold: {threshold:.3f}")
        
        # Additional state tracking info
        if current_state != "NOT_DRINKING":
            logger.info(f"  - Consecutive frames: {getattr(drinking_detector, 'consecutive_frames', 0)}")
            logger.info(f"  - Drinking frames count: {drinking_detector.drinking_frames_count}")
            logger.info(f"  - Min required drinking frames: {drinking_detector.min_drinking_frames}")
        
        # Add processing time to frames
        cv2.putText(
            posture_frame,
            f"Model: {'Heavy' if args.posture_model == 2 else 'Full' if args.posture_model == 1 else 'Lite'} ({posture_time:.3f}s)",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        
        cv2.putText(
            drinking_frame,
            f"Model: {'Heavy' if args.drinking_model == 2 else 'Full' if args.drinking_model == 1 else 'Lite'} ({drinking_time:.3f}s)",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        
        # Display frames
        cv2.imshow("Posture Detection", posture_frame)
        cv2.imshow("Drinking Detection", drinking_frame)
        
        # Log events
        if posture_event:
            logger.info(f"Posture event: {posture_event.state.name}, confidence: {posture_event.confidence:.2f}")
        
        if drinking_event:
            logger.info(f"Drinking event detected, confidence: {drinking_event.confidence:.2f}, duration: {drinking_event.duration:.2f}s")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
