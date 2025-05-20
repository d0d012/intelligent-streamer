"""
Analytics engine for video processing.
Combines object detection, motion analysis, and tracking.
"""

import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Any, Optional
from queue import Queue, Empty

from src.utils.config import get_config
from src.utils.logging import get_logger
from .detector import DetectorManager, DetectionResult
from .heatmap import MotionDetector

class AnalyticsResult:
    """
    Represents combined results from multiple analytics components.
    """
    
    def __init__(self, 
                 frame_id: int = 0,
                 timestamp: float = None):
        """
        Initialize analytics result.
        
        Args:
            frame_id: Frame identifier
            timestamp: Result timestamp (default: current time)
        """
        self.frame_id = frame_id
        self.timestamp = timestamp or time.time()
        self.detections = []
        self.detection_count = 0
        self.motion_level = 0.0
        self.motion_areas = []
        self.processing_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "detections": self.detections,
            "detection_count": self.detection_count,
            "motion_level": self.motion_level,
            "motion_areas": self.motion_areas
        }


class AnalyticsEngine:
    """
    Main analytics engine that coordinates detection and tracking.
    """
    
    def __init__(self):
        """Initialize the analytics engine."""
        self.logger = get_logger("analytics_engine")
        self.config = get_config()
        
        # Component managers
        self.detector_manager = DetectorManager()
        
        # Processing settings
        self.enabled = self.config.get("analytics", "enabled", True)
        self.max_queue_size = self.config.get("analytics", "max_queue_size", 10)
        
        # Status information
        self.sources = {}
        self.active = False
        
        # Processing thread
        self._thread = None
        self._stop_event = threading.Event()
        self._frame_queue = Queue(maxsize=self.max_queue_size)
        
        self.logger.info("Analytics engine initialized")
    
    def start(self) -> bool:
        """
        Start the analytics engine.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.active:
            self.logger.warning("Analytics engine already active")
            return True
        
        try:
            # Clear any existing items in the queue
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get(block=False)
                except Empty:
                    break
            
            # Start processing thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._process_frames)
            self._thread.daemon = True
            self._thread.start()
            
            self.active = True
            self.logger.info("Analytics engine started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting analytics engine: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the analytics engine.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.active:
            return True
        
        try:
            # Signal thread to stop
            self._stop_event.set()
            
            # Wait for thread to end
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3.0)
            
            self.active = False
            self.logger.info("Analytics engine stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping analytics engine: {e}")
            return False
    
    def add_source(self, source_id: str, width: int = 640, height: int = 480) -> bool:
        """
        Add a video source for analytics.
        
        Args:
            source_id: Unique source identifier
            width: Frame width
            height: Frame height
            
        Returns:
            True if added successfully, False otherwise
        """
        if source_id in self.sources:
            self.logger.warning(f"Source already exists: {source_id}")
            return False
        
        try:
            # Create source-specific components
            source_info = {
                "id": source_id,
                "width": width,
                "height": height,
                "enabled": self.enabled,
                "frame_count": 0,
                "last_result": None,
                "detector": self.detector_manager.get_detector(),
                "motion_detector": MotionDetector(width, height),
                "last_processed_time": 0
            }
            
            self.sources[source_id] = source_info
            self.logger.info(f"Added source for analytics: {source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding source: {e}")
            return False
    
    def remove_source(self, source_id: str) -> bool:
        """
        Remove a video source from analytics.
        
        Args:
            source_id: Source identifier
            
        Returns:
            True if removed successfully, False otherwise
        """
        if source_id not in self.sources:
            self.logger.warning(f"Source not found: {source_id}")
            return False
        
        try:
            # Remove source
            del self.sources[source_id]
            self.logger.info(f"Removed source from analytics: {source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing source: {e}")
            return False
    
    def toggle_source(self, source_id: str, enabled: bool) -> bool:
        """
        Enable or disable analytics for a source.
        
        Args:
            source_id: Source identifier
            enabled: Whether analytics should be enabled
            
        Returns:
            True if successful, False otherwise
        """
        if source_id not in self.sources:
            self.logger.warning(f"Source not found: {source_id}")
            return False
        
        try:
            self.sources[source_id]["enabled"] = enabled
            self.logger.info(f"Analytics for source {source_id} {'enabled' if enabled else 'disabled'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error toggling source: {e}")
            return False
    
    def process_frame(self, source_id: str, frame: np.ndarray, 
                     frame_id: int = 0) -> bool:
        """
        Queue a frame for processing.
        
        Args:
            source_id: Source identifier
            frame: BGR frame
            frame_id: Frame identifier (optional)
            
        Returns:
            True if frame was queued, False otherwise
        """
        if source_id not in self.sources:
            self.logger.warning(f"Source not found: {source_id}")
            return False
        
        if not self.active:
            self.logger.warning("Analytics engine not active")
            return False
        
        if not self.sources[source_id]["enabled"]:
            return False
        
        try:
            # Add to queue if not full
            if not self._frame_queue.full():
                item = {
                    "source_id": source_id,
                    "frame": frame.copy(),
                    "frame_id": frame_id,
                    "timestamp": time.time()
                }
                self._frame_queue.put(item, block=False)
                return True
            else:
                # Skip frame if queue is full
                return False
                
        except Exception as e:
            self.logger.error(f"Error queueing frame: {e}")
            return False
    
    def get_result(self, source_id: str) -> Optional[AnalyticsResult]:
        """
        Get latest result for a source.
        
        Args:
            source_id: Source identifier
            
        Returns:
            AnalyticsResult or None if not available
        """
        if source_id not in self.sources:
            return None
        
        return self.sources[source_id].get("last_result")
    
def annotate_frame(self, source_id: str, frame: np.ndarray) -> np.ndarray:
        """
        Annotate frame with analytics results.
        
        Args:
            source_id: Source identifier
            frame: BGR frame
            
        Returns:
            Annotated frame
        """
        if source_id not in self.sources:
            return frame.copy()
        
        # Get source info
        source_info = self.sources[source_id]
        
        # Create a copy of the frame
        result = frame.copy()
        
        # Get latest result
        analytics_result = source_info.get("last_result")
        
        if analytics_result:
            # Annotate with object detections
            if analytics_result.detections:
                # Create a temporary detection result for the detector to use
                detection_result = DetectionResult(analytics_result.detections)
                detection_result.processing_time = analytics_result.processing_time
                
                # Draw detections
                result = source_info["detector"].annotate_frame(result, detection_result)
            
            # Overlay motion heatmap
            if source_info["motion_detector"]:
                # Use alpha blending to overlay heatmap
                heatmap = source_info["motion_detector"].get_heatmap_overlay(result)
                
                # Add motion information
                motion_text = f"Motion: {analytics_result.motion_level:.1%}"
                cv2.putText(
                    result,
                    motion_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Draw motion areas
                for area in analytics_result.motion_areas:
                    x, y, w, h = area["x"], area["y"], area["width"], area["height"]
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        # Add info text
        cv2.putText(
            result,
            f"Analytics: {'ON' if source_info['enabled'] else 'OFF'}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            result,
            timestamp,
            (10, result.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return result
    
def _process_frames(self):
    """
    Process frames from the queue.
    This runs in a separate thread.
    """
    self.logger.debug("Analytics processing thread started")
    
    while not self._stop_event.is_set():
        try:
            # Get next frame from queue
            try:
                item = self._frame_queue.get(timeout=0.1)
            except Empty:
                # No frames to process
                continue
            
            # Extract frame information
            source_id = item["source_id"]
            frame = item["frame"]
            frame_id = item["frame_id"]
            timestamp = item["timestamp"]
            
            # Skip if source not found or disabled
            if (source_id not in self.sources or 
                not self.sources[source_id]["enabled"]):
                continue
            
            # Get source info
            source_info = self.sources[source_id]
            
            # Create result object
            result = AnalyticsResult(frame_id, timestamp)
            
            # Process with detector
            start_time = time.time()
            
            # Run object detection
            detection_result = source_info["detector"].detect(frame)
            
            # Copy detections
            result.detections = detection_result.detections
            result.detection_count = detection_result.count
            
            # Run motion detection
            motion_mask, motion_level, motion_areas = source_info["motion_detector"].detect(frame)
            
            # Copy motion results
            result.motion_level = motion_level
            result.motion_areas = motion_areas
            
            # Record processing time
            result.processing_time = time.time() - start_time
            
            # Store result
            source_info["last_result"] = result
            source_info["frame_count"] += 1
            source_info["last_processed_time"] = time.time()
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    self.logger.debug("Analytics processing thread stopped")

def get_status(self) -> Dict[str, Any]:
    """
    Get status information for the analytics engine.
    
    Returns:
        Dictionary with status information
    """
    status = {
        "active": self.active,
        "enabled": self.enabled,
        "sources": {},
        "queue_size": self._frame_queue.qsize(),
        "queue_full": self._frame_queue.full()
    }
    
    # Add source status
    for source_id, source_info in self.sources.items():
        source_status = {
            "enabled": source_info["enabled"],
            "frame_count": source_info["frame_count"],
            "last_processed_time": source_info["last_processed_time"]
        }
        
        # Add latest result summary if available
        if source_info.get("last_result"):
            result = source_info["last_result"]
            source_status.update({
                "detection_count": result.detection_count,
                "motion_level": result.motion_level,
                "processing_time": result.processing_time
            })
        
        status["sources"][source_id] = source_status
    
    return status

def shutdown(self):
    """Shut down the analytics engine."""
    self.logger.info("Shutting down analytics engine")
    self.stop()
    self.detector_manager.shutdown()
    self.sources.clear()
    self.logger.info("Analytics engine shut down")