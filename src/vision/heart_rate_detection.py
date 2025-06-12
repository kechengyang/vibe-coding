"""
Heart Rate Detection Module for Employee Health Monitoring System.

This module implements remote photoplethysmography (rPPG) to detect heart rate
from facial skin color changes captured by a webcam.
"""
import cv2
import time
import logging
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List, Optional, Deque
from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.fft import fft, fftfreq

logger = logging.getLogger("employee_health_monitor.vision.heart_rate_detection")

@dataclass
class HeartRateEvent:
    """Class for storing heart rate event data."""
    timestamp: float
    bpm: float
    confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage or serialization."""
        return {
            "timestamp": self.timestamp,
            "bpm": self.bpm,
            "confidence": self.confidence,
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))
        }


class HeartRateDetector:
    """Class for detecting heart rate using remote photoplethysmography (rPPG)."""
    
    def __init__(
        self,
        buffer_size: int = 300,  # 10 seconds at 30 fps
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        update_interval: int = 30,  # Update heart rate every 30 frames
        min_bpm: int = 45,
        max_bpm: int = 240,
        roi_color: Tuple[int, int, int] = (0, 255, 0),  # Green for ROI visualization
    ):
        """Initialize the heart rate detector.
        
        Args:
            buffer_size: Number of frames to keep in the signal buffer
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            update_interval: Number of frames between heart rate updates
            min_bpm: Minimum heart rate in beats per minute
            max_bpm: Maximum heart rate in beats per minute
            roi_color: Color to use for visualizing regions of interest
        """
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_faces=1,
        )
        
        # Configuration parameters
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.roi_color = roi_color
        
        # Signal buffers for RGB channels
        self.r_signal = deque(maxlen=buffer_size)
        self.g_signal = deque(maxlen=buffer_size)
        self.b_signal = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # ROI landmarks for forehead and cheeks
        # Using specific face mesh landmarks for these regions
        self.forehead_landmarks = [10, 67, 69, 104, 108, 151, 337, 338, 371, 9]
        self.left_cheek_landmarks = [205, 425, 429, 199, 208, 428]
        self.right_cheek_landmarks = [425, 205, 429, 199, 208, 428]
        
        # State variables
        self.current_bpm = 0.0
        self.confidence = 0.0
        self.frame_count = 0
        self.last_update_time = time.time()
        self.events = []
        
        logger.info(
            f"HeartRateDetector initialized with buffer_size={buffer_size}, "
            f"update_interval={update_interval}, bpm_range={min_bpm}-{max_bpm}"
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[HeartRateEvent]]:
        """Process a video frame to detect heart rate.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated frame, heart rate event if updated)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for face mesh
        results = self.face_mesh.process(rgb_frame)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Initialize event to None
        event = None
        
        # Extract signals if face is detected
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract ROIs and signals
            forehead_roi, forehead_mask = self._extract_roi(
                annotated_frame, face_landmarks, self.forehead_landmarks
            )
            left_cheek_roi, left_cheek_mask = self._extract_roi(
                annotated_frame, face_landmarks, self.left_cheek_landmarks
            )
            right_cheek_roi, right_cheek_mask = self._extract_roi(
                annotated_frame, face_landmarks, self.right_cheek_landmarks
            )
            
            # Combine ROIs
            combined_roi = np.vstack([forehead_roi, left_cheek_roi, right_cheek_roi])
            
            # Extract average RGB values from ROIs
            r_mean = np.mean(combined_roi[:, 2])  # OpenCV uses BGR
            g_mean = np.mean(combined_roi[:, 1])
            b_mean = np.mean(combined_roi[:, 0])
            
            # Add to signal buffers
            self.r_signal.append(r_mean)
            self.g_signal.append(g_mean)
            self.b_signal.append(b_mean)
            self.timestamps.append(time.time())
            
            # Increment frame count
            self.frame_count += 1
            
            # Update heart rate periodically
            if self.frame_count % self.update_interval == 0 and len(self.r_signal) >= self.buffer_size * 0.9:
                # Calculate heart rate
                self._calculate_heart_rate()
                
                # Create event
                event = HeartRateEvent(
                    timestamp=time.time(),
                    bpm=self.current_bpm,
                    confidence=self.confidence
                )
                
                # Record event
                self.events.append(event)
                
                logger.info(f"Heart rate updated: {self.current_bpm:.1f} BPM (confidence: {self.confidence:.2f})")
            
            # Draw face mesh landmarks
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Add heart rate information to top-left corner with smaller font
        confidence_color = (0, 255, 0) if self.confidence >= 0.5 else (0, 165, 255) if self.confidence >= 0.3 else (0, 0, 255)
        
        # Get frame dimensions
        height, width, _ = annotated_frame.shape
        
        # Add semi-transparent background for better readability
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (200, 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        
        # Add section title
        cv2.putText(
            annotated_frame,
            "HEART RATE",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Add heart rate text
        cv2.putText(
            annotated_frame,
            f"Rate: {self.current_bpm:.1f} BPM",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            confidence_color,
            1
        )
        
        # Add confidence text
        cv2.putText(
            annotated_frame,
            f"Confidence: {self.confidence:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            confidence_color,
            1
        )
        
        return annotated_frame, event
    
    def _extract_roi(
        self, 
        frame: np.ndarray, 
        face_landmarks, 
        landmark_indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract region of interest from the frame based on face landmarks.
        
        Args:
            frame: Input video frame
            face_landmarks: MediaPipe face landmarks
            landmark_indices: Indices of landmarks defining the ROI
            
        Returns:
            Tuple of (ROI pixels, ROI mask)
        """
        height, width, _ = frame.shape
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Extract landmark points
        points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
        
        # Draw filled polygon on mask
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        
        # Draw ROI on frame for visualization
        cv2.polylines(frame, [points_array], True, self.roi_color, 1)
        
        # Extract ROI pixels
        roi_pixels = frame[mask > 0]
        
        return roi_pixels, mask
    
    def _calculate_heart_rate(self) -> None:
        """Calculate heart rate from the collected signal buffers."""
        # Check if we have enough data
        if len(self.r_signal) < self.buffer_size * 0.5:
            self.current_bpm = 0.0
            self.confidence = 0.0
            return
        
        # Convert deques to numpy arrays
        r_signal = np.array(self.r_signal)
        g_signal = np.array(self.g_signal)
        b_signal = np.array(self.b_signal)
        timestamps = np.array(self.timestamps)
        
        # Calculate average sampling rate
        sampling_rate = len(timestamps) / (timestamps[-1] - timestamps[0])
        
        # Normalize signals
        r_normalized = self._normalize_signal(r_signal)
        g_normalized = self._normalize_signal(g_signal)
        b_normalized = self._normalize_signal(b_signal)
        
        # Green channel typically has the strongest plethysmographic signal
        # but we can also use a combination of channels
        # For simplicity, we'll use the green channel
        raw_signal = g_normalized
        
        # Apply bandpass filter to isolate frequencies in the range of human heart rates
        # 0.75 Hz - 4 Hz corresponds to 45-240 BPM
        filtered_signal = self._bandpass_filter(raw_signal, sampling_rate, 0.75, 4.0)
        
        # Perform FFT to find dominant frequency
        bpm, signal_strength = self._find_heart_rate(filtered_signal, sampling_rate)
        
        # Update state if the result is reasonable
        if self.min_bpm <= bpm <= self.max_bpm:
            # Smooth the result with previous value if available
            if self.current_bpm > 0:
                self.current_bpm = 0.7 * self.current_bpm + 0.3 * bpm
            else:
                self.current_bpm = bpm
            
            # Calculate confidence based on signal strength and stability
            self.confidence = min(1.0, signal_strength * 2.0)
        
        logger.debug(f"Raw BPM: {bpm:.1f}, Signal strength: {signal_strength:.2f}")
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance.
        
        Args:
            signal: Input signal
            
        Returns:
            Normalized signal
        """
        # Remove mean (detrend)
        detrended = signal - np.mean(signal)
        
        # Normalize
        std = np.std(detrended)
        if std > 0:
            normalized = detrended / std
        else:
            normalized = detrended
        
        return normalized
    
    def _bandpass_filter(
        self, 
        signal_data: np.ndarray, 
        sampling_rate: float, 
        low_freq: float, 
        high_freq: float
    ) -> np.ndarray:
        """Apply simple bandpass filter to signal.
        
        Args:
            signal_data: Input signal
            sampling_rate: Sampling rate in Hz
            low_freq: Lower cutoff frequency in Hz
            high_freq: Upper cutoff frequency in Hz
            
        Returns:
            Filtered signal
        """
        # Get signal length
        n = len(signal_data)
        
        # Compute FFT
        fft_result = fft(signal_data)
        freq = fftfreq(n, 1/sampling_rate)
        
        # Create bandpass filter mask
        mask = np.zeros(n)
        for i in range(n):
            if abs(freq[i]) >= low_freq and abs(freq[i]) <= high_freq:
                mask[i] = 1
        
        # Apply filter in frequency domain
        filtered_fft = fft_result * mask
        
        # Convert back to time domain
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_signal
    
    def _find_heart_rate(self, signal: np.ndarray, sampling_rate: float) -> Tuple[float, float]:
        """Find heart rate using FFT.
        
        Args:
            signal: Input signal
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (heart rate in BPM, signal strength)
        """
        # Apply window function to reduce spectral leakage
        windowed_signal = signal * np.hamming(len(signal))
        
        # Compute FFT
        n = len(windowed_signal)
        fft_result = fft(windowed_signal)
        magnitude = np.abs(fft_result[:n//2])
        
        # Get frequencies
        freq = fftfreq(n, 1/sampling_rate)[:n//2]
        
        # Find peaks in the frequency range corresponding to heart rates
        min_freq = self.min_bpm / 60.0  # Convert BPM to Hz
        max_freq = self.max_bpm / 60.0
        
        # Find indices within the valid frequency range
        valid_indices = np.where((freq >= min_freq) & (freq <= max_freq))[0]
        
        if len(valid_indices) == 0:
            return 0.0, 0.0
        
        # Find the peak frequency
        peak_idx = valid_indices[np.argmax(magnitude[valid_indices])]
        peak_freq = freq[peak_idx]
        
        # Convert frequency to BPM
        bpm = peak_freq * 60.0
        
        # Calculate signal strength (normalized peak magnitude)
        signal_strength = magnitude[peak_idx] / np.sum(magnitude)
        
        return bpm, signal_strength
    
    def get_current_bpm(self) -> float:
        """Get the current heart rate in beats per minute.
        
        Returns:
            float: Current heart rate in BPM
        """
        return self.current_bpm
    
    def get_confidence(self) -> float:
        """Get the confidence level of the heart rate measurement.
        
        Returns:
            float: Confidence level (0-1)
        """
        return self.confidence
    
    def get_events(self) -> List[HeartRateEvent]:
        """Get all recorded heart rate events.
        
        Returns:
            List[HeartRateEvent]: List of heart rate events
        """
        return self.events.copy()
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.r_signal.clear()
        self.g_signal.clear()
        self.b_signal.clear()
        self.timestamps.clear()
        self.current_bpm = 0.0
        self.confidence = 0.0
        self.frame_count = 0
        self.events = []
        
        logger.info("HeartRateDetector reset")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create video capture
    cap = cv2.VideoCapture(0)
    
    # Create heart rate detector
    detector = HeartRateDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, event = detector.process_frame(frame)
        
        # Display frame
        cv2.imshow("Heart Rate Detection", annotated_frame)
        
        # Print event if heart rate updated
        if event:
            print(f"Heart rate: {event.bpm:.1f} BPM (confidence: {event.confidence:.2f})")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
