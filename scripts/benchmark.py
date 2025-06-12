#!/usr/bin/env python3
"""
Benchmark script for the Employee Health Monitoring System.

This script benchmarks the performance of the computer vision components.
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.vision.capture import VideoCapture
from src.vision.pose_detection import PostureDetector
from src.vision.action_detection import DrinkingDetector


def benchmark_video_capture(video_path: str, num_frames: int = 100) -> float:
    """Benchmark the VideoCapture class.
    
    Args:
        video_path: Path to a video file for benchmarking
        num_frames: Number of frames to process
        
    Returns:
        float: Average FPS
    """
    print(f"Benchmarking VideoCapture with {num_frames} frames...")
    
    # Create video capture
    cap = cv2.VideoCapture(video_path)
    
    # Process frames
    start_time = time.time()
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
    
    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    
    # Clean up
    cap.release()
    
    print(f"VideoCapture: {fps:.2f} FPS")
    return fps


def benchmark_posture_detector(video_path: str, num_frames: int = 100) -> float:
    """Benchmark the PostureDetector class.
    
    Args:
        video_path: Path to a video file for benchmarking
        num_frames: Number of frames to process
        
    Returns:
        float: Average FPS
    """
    print(f"Benchmarking PostureDetector with {num_frames} frames...")
    
    # Create video capture and posture detector
    cap = cv2.VideoCapture(video_path)
    detector = PostureDetector()
    
    # Process frames
    start_time = time.time()
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detector.process_frame(frame)
    
    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    
    # Clean up
    cap.release()
    
    print(f"PostureDetector: {fps:.2f} FPS")
    return fps


def benchmark_drinking_detector(video_path: str, num_frames: int = 100) -> float:
    """Benchmark the DrinkingDetector class.
    
    Args:
        video_path: Path to a video file for benchmarking
        num_frames: Number of frames to process
        
    Returns:
        float: Average FPS
    """
    print(f"Benchmarking DrinkingDetector with {num_frames} frames...")
    
    # Create video capture and drinking detector
    cap = cv2.VideoCapture(video_path)
    detector = DrinkingDetector()
    
    # Process frames
    start_time = time.time()
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detector.process_frame(frame)
    
    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    
    # Clean up
    cap.release()
    
    print(f"DrinkingDetector: {fps:.2f} FPS")
    return fps


def benchmark_full_pipeline(video_path: str, num_frames: int = 100) -> float:
    """Benchmark the full computer vision pipeline.
    
    Args:
        video_path: Path to a video file for benchmarking
        num_frames: Number of frames to process
        
    Returns:
        float: Average FPS
    """
    print(f"Benchmarking full pipeline with {num_frames} frames...")
    
    # Create video capture and detectors
    cap = cv2.VideoCapture(video_path)
    posture_detector = PostureDetector()
    drinking_detector = DrinkingDetector()
    
    # Process frames
    start_time = time.time()
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with posture detector
        posture_frame, _ = posture_detector.process_frame(frame)
        
        # Process frame with drinking detector
        drinking_detector.process_frame(posture_frame)
    
    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    
    # Clean up
    cap.release()
    
    print(f"Full pipeline: {fps:.2f} FPS")
    return fps


def generate_test_video(output_path: str, duration: int = 10, fps: int = 30, resolution: tuple = (640, 480)) -> str:
    """Generate a test video for benchmarking.
    
    Args:
        output_path: Path to save the test video
        duration: Duration of the video in seconds
        fps: Frames per second
        resolution: Video resolution as (width, height)
        
    Returns:
        str: Path to the generated video
    """
    print(f"Generating test video: {duration}s, {fps} FPS, {resolution[0]}x{resolution[1]}...")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    # Generate frames
    num_frames = duration * fps
    for i in range(num_frames):
        # Create a frame with a moving rectangle
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Calculate rectangle position
        x = int((i / num_frames) * (resolution[0] - 100))
        y = int(resolution[1] / 2 - 50)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
        
        # Add frame number
        cv2.putText(
            frame,
            f"Frame: {i}/{num_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Write frame
        out.write(frame)
    
    # Clean up
    out.release()
    
    print(f"Test video generated: {output_path}")
    return output_path


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark the Employee Health Monitoring System")
    parser.add_argument("--video", help="Path to a video file for benchmarking")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--generate", action="store_true", help="Generate a test video")
    parser.add_argument("--duration", type=int, default=10, help="Duration of the test video in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the test video")
    parser.add_argument("--width", type=int, default=640, help="Width of the test video")
    parser.add_argument("--height", type=int, default=480, help="Height of the test video")
    args = parser.parse_args()
    
    # Generate test video if requested
    if args.generate:
        video_path = os.path.join(project_root, "test_video.avi")
        generate_test_video(
            video_path,
            duration=args.duration,
            fps=args.fps,
            resolution=(args.width, args.height)
        )
        if not args.video:
            args.video = video_path
    
    # Check if video path is provided
    if not args.video:
        print("Error: No video path provided. Use --video or --generate.")
        return 1
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    print("=" * 50)
    
    video_capture_fps = benchmark_video_capture(args.video, args.frames)
    print("-" * 50)
    
    posture_detector_fps = benchmark_posture_detector(args.video, args.frames)
    print("-" * 50)
    
    drinking_detector_fps = benchmark_drinking_detector(args.video, args.frames)
    print("-" * 50)
    
    full_pipeline_fps = benchmark_full_pipeline(args.video, args.frames)
    print("=" * 50)
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Video Capture:     {video_capture_fps:.2f} FPS")
    print(f"Posture Detector:  {posture_detector_fps:.2f} FPS")
    print(f"Drinking Detector: {drinking_detector_fps:.2f} FPS")
    print(f"Full Pipeline:     {full_pipeline_fps:.2f} FPS")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
