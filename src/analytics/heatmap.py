"""
Motion heatmap generation for video analytics.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional

from src.utils.logging import get_logger

class MotionHeatmap:
    """
    Generates and maintains a motion heatmap for video frames.
    Identifies areas of persistent motion over time.
    """
    
    def __init__(self, 
                frame_width: int = 640, 
                frame_height: int = 480,
                decay_factor: float = 0.95,
                threshold: int = 25):
        """
        Initialize motion heatmap.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            decay_factor: Decay factor for heatmap (0-1)
            threshold: Threshold for motion detection
        """
        self.width = frame_width
        self.height = frame_height
        self.decay_factor = decay_factor
        self.threshold = threshold
        
        # Initialize heatmap
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Previous frame for difference calculation
        self.prev_frame = None
        
        # Logger
        self.logger = get_logger("motion_heatmap")
        self.logger.info(f"Motion heatmap initialized ({frame_width}x{frame_height})")
    
    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Update heatmap with a new frame.
        
        Args:
            frame: BGR frame
            
        Returns:
            Motion mask for the current frame
        """
        # Convert to grayscale
        if frame.shape[2] == 3:  # Make sure it has 3 channels
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Resize if needed
        if gray.shape[1] != self.width or gray.shape[0] != self.height:
            gray = cv2.resize(gray, (self.width, self.height))
        
        # Initialize previous frame if needed
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Compute absolute difference
        frame_diff = cv2.absdiff(gray, self.prev_frame)
        
        # Apply threshold to get binary motion mask
        _, motion_mask = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Apply some morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        # Update heatmap: decay existing values and add new motion
        self.heatmap = self.heatmap * self.decay_factor
        self.heatmap = np.maximum(self.heatmap, motion_mask.astype(np.float32) / 255.0)
        
        # Update previous frame
        self.prev_frame = gray
        
        return motion_mask
    
    def get_heatmap(self) -> np.ndarray:
        """
        Get current heatmap as normalized 8-bit image.
        
        Returns:
            Heatmap as 8-bit grayscale image
        """
        # Normalize and convert to 8-bit
        normalized = np.clip(self.heatmap * 255, 0, 255).astype(np.uint8)
        return normalized
    
    def get_colored_heatmap(self) -> np.ndarray:
        """
        Get colored visualization of the heatmap.
        
        Returns:
            Colored heatmap visualization
        """
        # Get normalized grayscale heatmap
        normalized = self.get_heatmap()
        
        # Apply color map
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored
    
    def get_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on the original frame.
        
        Args:
            frame: Original BGR frame
            alpha: Opacity of the overlay (0-1)
            
        Returns:
            Frame with heatmap overlay
        """
        # Resize frame if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Get colored heatmap
        heatmap_colored = self.get_colored_heatmap()
        
        # Create mask for non-zero regions
        mask = self.get_heatmap() > 10
        
        # Create result image
        result = frame.copy()
        
        # Apply overlay only in mask regions
        cv2.addWeighted(
            heatmap_colored, 
            alpha, 
            frame, 
            1 - alpha, 
            0, 
            dst=result, 
            mask=mask.astype(np.uint8) * 255
        )
        
        return result
    
    def reset(self):
        """Reset the heatmap to zero."""
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.prev_frame = None
        self.logger.info("Motion heatmap reset")


class MotionDetector:
    """
    Detects and analyzes motion in video frames.
    Provides more detailed analysis beyond basic heatmap.
    """
    
    def __init__(self, 
                frame_width: int = 640, 
                frame_height: int = 480):
        """
        Initialize motion detector.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.width = frame_width
        self.height = frame_height
        
        # Create motion heatmap
        self.heatmap = MotionHeatmap(frame_width, frame_height)
        
        # MOG2 background subtractor for better motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
        
        # Logger
        self.logger = get_logger("motion_detector")
        self.logger.info(f"Motion detector initialized ({frame_width}x{frame_height})")
        
        # Motion statistics
        self.motion_level = 0.0
        self.motion_areas = []
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, float, list]:
        """
        Detect motion in a frame.
        
        Args:
            frame: BGR frame
            
        Returns:
            Tuple of (motion mask, motion level, motion areas)
        """
        # Apply background subtractor
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Update heatmap
        self.heatmap.update(frame)
        
        # Calculate motion level (percentage of frame with motion)
        self.motion_level = np.sum(fg_mask > 0) / (self.width * self.height)
        
        # Identify motion areas (contours)
        contours, _ = cv2.findContours(
            fg_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process contours to get bounding rectangles
        self.motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                self.motion_areas.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": area
                })
        
        return fg_mask, self.motion_level, self.motion_areas
    
    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotate frame with motion information.
        
        Args:
            frame: BGR frame
            
        Returns:
            Annotated frame
        """
        # Create a copy of the frame
        result = frame.copy()
        
        # Draw motion areas
        for area in self.motion_areas:
            x, y, w, h = area["x"], area["y"], area["width"], area["height"]
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Add motion level text
        cv2.putText(
            result,
            f"Motion: {self.motion_level:.1%}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        return result
    
    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Get frame with heatmap overlay.
        
        Args:
            frame: BGR frame
            
        Returns:
            Frame with heatmap overlay
        """
        return self.heatmap.get_overlay(frame)
    
    def reset(self):
        """Reset detector state."""
        self.heatmap.reset()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
        self.motion_level = 0.0
        self.motion_areas = []
        self.logger.info("Motion detector reset")