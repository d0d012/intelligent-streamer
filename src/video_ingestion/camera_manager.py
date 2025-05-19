"""
Camera manager for handling multiple video sources.
"""

import cv2
import threading
import time
import os
from typing import Dict, List, Tuple, Any, Optional
import uuid

from src.utils.config import get_config
from src.utils.logging import get_logger

# Import pipeline implementations
from .pipelines.rtsp import RtspPipeline
from .pipelines.udp import UdpPipeline

class VideoSource:
    """
    Represents a single video source (camera, RTSP stream, etc.)
    and manages its capture, processing, and streaming.
    """
    
    def __init__(
        self,
        source_id: str,
        source: Any,  # Camera index (int) or URL (str)
        resolution: Tuple[int, int] = (640, 480),
        framerate: int = 30
    ):
        """
        Initialize a video source.
        
        Args:
            source_id: Unique identifier for this source
            source: OpenCV camera index (int) or video source URL (str)
            resolution: Frame resolution as (width, height)
            framerate: Target capture framerate
        """
        self.source_id = source_id
        self.source = source
        self.resolution = resolution
        self.framerate = framerate
        
        # Configure logger
        self.logger = get_logger(f"camera.{source_id}")
        
        # Get configuration
        config = get_config()
        
        # State variables
        self.active = False
        self.capture = None
        self.frame_count = 0
        self.fps = 0.0
        self.last_frame = None
        self.last_frame_time = 0
        
        # Create pipelines for streaming
        streaming_config = config.get_section("streaming")
        
        # Pipeline for raw frames
        self.raw_pipelines = {}
        self.annotated_pipelines = {}
        
        # Settings
        self.analytics_enabled = config.get("analytics", "enabled", True)
        
        # Processing thread
        self._thread = None
        self._stop_event = threading.Event()
    
    def add_raw_pipeline(self, name: str, pipeline: Any) -> bool:
        """
        Add a pipeline for streaming raw frames.
        
        Args:
            name: Pipeline name
            pipeline: Pipeline instance
            
        Returns:
            True if added successfully, False otherwise
        """
        if name in self.raw_pipelines:
            self.logger.warning(f"Raw pipeline {name} already exists")
            return False
        
        self.raw_pipelines[name] = pipeline
        return True
    
    def add_annotated_pipeline(self, name: str, pipeline: Any) -> bool:
        """
        Add a pipeline for streaming annotated frames.
        
        Args:
            name: Pipeline name
            pipeline: Pipeline instance
            
        Returns:
            True if added successfully, False otherwise
        """
        if name in self.annotated_pipelines:
            self.logger.warning(f"Annotated pipeline {name} already exists")
            return False
        
        self.annotated_pipelines[name] = pipeline
        return True
    
    def start(self) -> bool:
        """
        Start capturing from this source.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.active:
            self.logger.warning("Source is already active")
            return True
        
        try:
            # Initialize capture
            self.logger.info(f"Starting capture from {self.source}")
            self.capture = cv2.VideoCapture(self.source)
            
            # Configure resolution and framerate
            width, height = self.resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, self.framerate)
            
            # Check if capture is successful
            if not self.capture.isOpened():
                self.logger.error(f"Failed to open capture: {self.source}")
                return False
            
            # Reset counters
            self.frame_count = 0
            self.fps = 0.0
            
            # Start pipelines
            for name, pipeline in self.raw_pipelines.items():
                if not pipeline.start():
                    self.logger.error(f"Failed to start raw pipeline {name}")
            
            for name, pipeline in self.annotated_pipelines.items():
                if not pipeline.start():
                    self.logger.error(f"Failed to start annotated pipeline {name}")
            
            # Start processing thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._process_frames)
            self._thread.daemon = True
            self._thread.start()
            
            self.active = True
            self.logger.info(f"Source started successfully: {self.source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting source: {e}")
            self._cleanup()
            return False
    
    def stop(self) -> bool:
        """
        Stop capturing from this source.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.active:
            return True
        
        self.logger.info(f"Stopping source: {self.source_id}")
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to end
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        
        # Clean up resources
        self._cleanup()
        
        self.active = False
        self.logger.info(f"Source stopped successfully: {self.source_id}")
        return True
    
    def _cleanup(self):
        """Clean up resources."""
        # Release capture
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Stop pipelines
        for name, pipeline in self.raw_pipelines.items():
            if not pipeline.stop():
                self.logger.error(f"Failed to stop raw pipeline {name}")
        
        for name, pipeline in self.annotated_pipelines.items():
            if not pipeline.stop():
                self.logger.error(f"Failed to stop annotated pipeline {name}")
    
    def _process_frames(self):
        """
        Process frames from the source.
        This runs in a separate thread.
        """
        self.logger.debug("Frame processing thread started")
        
        # Timing variables
        start_time = time.time()
        frame_count = 0
        
        while not self._stop_event.is_set():
            # Capture frame
            ret, frame = self.capture.read()
            
            if not ret:
                self.logger.warning("Failed to capture frame")
                # Small delay to avoid CPU spinning on error
                time.sleep(0.1)
                continue
            
            # Store captured frame
            self.last_frame = frame
            self.last_frame_time = time.time()
            
            # Send raw frame to raw pipelines
            frame_bytes = frame.tobytes()
            
            for name, pipeline in self.raw_pipelines.items():
                if not pipeline.send_frame(frame_bytes):
                    self.logger.warning(f"Failed to send frame to raw pipeline {name}")
            
            # Process frame for analytics if enabled
            if self.analytics_enabled:
                # In a real implementation, we'd do object detection here
                # For now, just add a timestamp
                processed_frame = frame.copy()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    processed_frame, 
                    f"{timestamp} - {self.source_id}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Send processed frame to annotated pipelines
                processed_bytes = processed_frame.tobytes()
                
                for name, pipeline in self.annotated_pipelines.items():
                    if not pipeline.send_frame(processed_bytes):
                        self.logger.warning(f"Failed to send frame to annotated pipeline {name}")
            
            # Update frame count
            self.frame_count += 1
            frame_count += 1
            
            # Calculate FPS every second
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                # Reset counters
                frame_count = 0
                start_time = current_time
                
                # Log FPS
                self.logger.debug(f"Source {self.source_id} FPS: {self.fps:.2f}")
        
        self.logger.debug("Frame processing thread stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about this source.
        
        Returns:
            Dictionary with status information
        """
        return {
            "source_id": self.source_id,
            "active": self.active,
            "resolution": self.resolution,
            "framerate": self.framerate,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "analytics_enabled": self.analytics_enabled,
            "last_frame_time": self.last_frame_time
        }
    
    def toggle_analytics(self, enabled: bool) -> bool:
        """
        Enable or disable analytics processing.
        
        Args:
            enabled: Whether analytics should be enabled
            
        Returns:
            True if successful, False otherwise
        """
        self.analytics_enabled = enabled
        self.logger.info(f"Analytics {'enabled' if enabled else 'disabled'} for {self.source_id}")
        return True
    
    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update source settings.
        
        Args:
            settings: Dictionary with settings to update
            
        Returns:
            True if successful, False otherwise
        """
        # Check if source is active
        was_active = self.active
        
        # Stop if active
        if was_active:
            self.stop()
        
        # Update settings
        if "resolution" in settings:
            self.resolution = settings["resolution"]
            self.logger.info(f"Updated resolution to {self.resolution}")
        
        if "framerate" in settings:
            self.framerate = settings["framerate"]
            self.logger.info(f"Updated framerate to {self.framerate}")
        
        # Restart if it was active
        if was_active:
            return self.start()
        
        return True


class CameraManager:
    """
    Manages multiple video sources and coordinates their operations.
    Focused on RTSP streaming for video distribution.
    """
    
    def __init__(self):
        """Initialize the camera manager."""
        self.sources = {}
        self.logger = get_logger("camera_manager")
        
        # Get configuration
        self.config = get_config()
        self.video_config = self.config.get_section("video_ingestion")
        self.streaming_config = self.config.get_section("streaming")
        
        # Default settings
        self.default_resolution = tuple(self.video_config.get(
            "default_resolution", [640, 480]
        ))
        self.default_framerate = self.video_config.get("default_framerate", 30)
        self.gstreamer_path = self.video_config.get("gstreamer_path", "gst-launch-1.0")
        
        # Streaming settings
        self.base_port = self.streaming_config.get("base_port", 8554)  # Default RTSP port range
        self.next_port = self.base_port
        
        self.logger.info("Camera manager initialized with RTSP streaming")
    
    def add_source(
        self,
        source: Any,
        source_id: str = None,
        resolution: Tuple[int, int] = None,
        framerate: int = None
    ) -> Optional[str]:
        """
        Add a new video source with RTSP streaming.
        
        Args:
            source: Camera index (int) or video source URL (str)
            source_id: Unique identifier (generated if None)
            resolution: Frame resolution (default from config)
            framerate: Target framerate (default from config)
            
        Returns:
            Source ID if added successfully, None otherwise
        """
        # Generate source ID if not provided
        if source_id is None:
            source_id = f"source_{uuid.uuid4().hex[:8]}"
        
        # Check if source ID already exists
        if source_id in self.sources:
            self.logger.error(f"Source ID already exists: {source_id}")
            return None
        
        # Use default values if not provided
        if resolution is None:
            resolution = self.default_resolution
        
        if framerate is None:
            framerate = self.default_framerate
        
        try:
            # Create video source
            video_source = VideoSource(
                source_id=source_id,
                source=source,
                resolution=resolution,
                framerate=framerate
            )
            
            # Add RTSP streaming pipelines
            # Raw RTSP pipeline
            raw_rtsp_port = self._get_next_port()
            raw_rtsp_pipeline = RtspPipeline(
                source_id=source_id,
                width=resolution[0],
                height=resolution[1],
                framerate=framerate,
                port=raw_rtsp_port,
                path=f"/{source_id}/raw",
                gst_path=self.gstreamer_path
            )
            video_source.add_raw_pipeline("rtsp", raw_rtsp_pipeline)
            
            # Annotated RTSP pipeline
            annotated_rtsp_port = self._get_next_port()
            annotated_rtsp_pipeline = RtspPipeline(
                source_id=source_id,
                width=resolution[0],
                height=resolution[1],
                framerate=framerate,
                port=annotated_rtsp_port,
                path=f"/{source_id}/annotated",
                gst_path=self.gstreamer_path
            )
            video_source.add_annotated_pipeline("rtsp", annotated_rtsp_pipeline)
            
            # Store the source
            self.sources[source_id] = video_source
            
            # Log the RTSP URLs for convenience
            self.logger.info(f"Added source {source_id} with RTSP streams:")
            self.logger.info(f"  - Raw: rtsp://localhost:{raw_rtsp_port}/{source_id}/raw")
            self.logger.info(f"  - Annotated: rtsp://localhost:{annotated_rtsp_port}/{source_id}/annotated")
            
            return source_id
            
        except Exception as e:
            self.logger.error(f"Error adding source: {e}")
            return None
    
    def remove_source(self, source_id: str) -> bool:
        """
        Remove a video source.
        
        Args:
            source_id: ID of the source to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if source_id not in self.sources:
            self.logger.warning(f"Source not found: {source_id}")
            return False
        
        # Stop the source if it's active
        if self.sources[source_id].active:
            self.sources[source_id].stop()
        
        # Remove from collection
        del self.sources[source_id]
        
        self.logger.info(f"Removed source: {source_id}")
        return True
    
    def start_source(self, source_id: str) -> bool:
        """
        Start a video source.
        
        Args:
            source_id: ID of the source to start
            
        Returns:
            True if started successfully, False otherwise
        """
        if source_id not in self.sources:
            self.logger.warning(f"Source not found: {source_id}")
            return False
        
        return self.sources[source_id].start()
    
    def stop_source(self, source_id: str) -> bool:
        """
        Stop a video source.
        
        Args:
            source_id: ID of the source to stop
            
        Returns:
            True if stopped successfully, False otherwise
        """
        if source_id not in self.sources:
            self.logger.warning(f"Source not found: {source_id}")
            return False
        
        return self.sources[source_id].stop()
    
    def start_all(self) -> bool:
        """
        Start all video sources.
        
        Returns:
            True if all sources started successfully, False otherwise
        """
        success = True
        
        for source_id, source in self.sources.items():
            if not source.start():
                self.logger.error(f"Failed to start source: {source_id}")
                success = False
        
        return success
    
    def stop_all(self) -> bool:
        """
        Stop all video sources.
        
        Returns:
            True if all sources stopped successfully, False otherwise
        """
        success = True
        
        for source_id, source in self.sources.items():
            if not source.stop():
                self.logger.error(f"Failed to stop source: {source_id}")
                success = False
        
        return success
    
    def get_source(self, source_id: str) -> Optional[VideoSource]:
        """
        Get a video source by ID.
        
        Args:
            source_id: ID of the source to get
            
        Returns:
            VideoSource instance or None if not found
        """
        return self.sources.get(source_id)
    
    def get_source_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a video source.
        
        Args:
            source_id: ID of the source
            
        Returns:
            Status dictionary or None if source not found
        """
        source = self.get_source(source_id)
        if source:
            return source.get_status()
        return None
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all video sources.
        
        Returns:
            Dictionary mapping source IDs to status dictionaries
        """
        return {
            source_id: source.get_status()
            for source_id, source in self.sources.items()
        }
    
    def get_rtsp_urls(self, source_id: str = None) -> Dict[str, Dict[str, str]]:
        """
        Get RTSP URLs for all sources or a specific source.
        
        Args:
            source_id: Optional ID of a specific source
            
        Returns:
            Dictionary with RTSP URLs by source and stream type
        """
        result = {}
        
        # Function to get URLs for a single source
        def get_source_urls(src_id, source):
            urls = {"raw": None, "annotated": None}
            
            if "rtsp" in source.raw_pipelines:
                pipeline = source.raw_pipelines["rtsp"]
                urls["raw"] = f"rtsp://localhost:{pipeline.port}{pipeline.path}"
            
            if "rtsp" in source.annotated_pipelines:
                pipeline = source.annotated_pipelines["rtsp"]
                urls["annotated"] = f"rtsp://localhost:{pipeline.port}{pipeline.path}"
            
            return urls
        
        # Get URLs for a specific source or all sources
        if source_id:
            source = self.get_source(source_id)
            if source:
                result[source_id] = get_source_urls(source_id, source)
        else:
            for src_id, source in self.sources.items():
                result[src_id] = get_source_urls(src_id, source)
        
        return result
    
    def toggle_analytics(self, source_id: str, enabled: bool) -> bool:
        """
        Enable or disable analytics for a source.
        
        Args:
            source_id: ID of the source
            enabled: Whether analytics should be enabled
            
        Returns:
            True if successful, False otherwise
        """
        source = self.get_source(source_id)
        if source:
            return source.toggle_analytics(enabled)
        return False
    
    def update_source_settings(self, source_id: str, settings: Dict[str, Any]) -> bool:
        """
        Update settings for a source.
        
        Args:
            source_id: ID of the source
            settings: Dictionary with settings to update
            
        Returns:
            True if successful, False otherwise
        """
        source = self.get_source(source_id)
        if source:
            return source.update_settings(settings)
        return False
    
    def _get_next_port(self) -> int:
        """
        Get the next available port for RTSP streaming.
        
        Returns:
            Port number
        """
        port = self.next_port
        self.next_port += 1
        return port
    
    def shutdown(self):
        """Shut down the camera manager and all sources."""
        self.logger.info("Shutting down camera manager")
        self.stop_all()
        self.sources.clear()
        self.logger.info("Camera manager shut down successfully")