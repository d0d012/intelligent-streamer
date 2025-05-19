"""
Base class for GStreamer pipeline implementations.
"""

import logging
import subprocess
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logging import get_logger

class GStreamerPipeline(ABC):
    """
    Abstract base class for GStreamer pipelines.
    Provides common functionality for different pipeline types.
    """
    
    def __init__(self, source_id: str, gst_path: str = "gst-launch-1.0"):
        """
        Initialize pipeline.
        
        Args:
            source_id: Unique identifier for the source
            gst_path: Path to GStreamer executable
        """
        self.source_id = source_id
        self.gst_path = gst_path
        self.process = None
        self.active = False
        self.logger = get_logger(f"pipeline.{source_id}")
        
    @abstractmethod
    def build_pipeline_args(self) -> List[str]:
        """
        Build the GStreamer pipeline arguments.
        
        Returns:
            List of command-line arguments for GStreamer
        """
        pass
    
    def start(self) -> bool:
        """
        Start the GStreamer pipeline.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.active:
            self.logger.warning("Pipeline is already active")
            return True
        
        try:
            # Build pipeline arguments
            args = self.build_pipeline_args()
            
            self.logger.debug(f"Starting pipeline with args: {' '.join(args)}")
            
            # Start the process
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.active = True
            self.logger.info(f"Pipeline started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the GStreamer pipeline.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.active:
            return True
        
        try:
            if self.process:
                # Close stdin to signal we're done sending data
                if self.process.stdin:
                    self.process.stdin.close()
                
                # Terminate the process
                self.process.terminate()
                
                # Wait for process to end (with timeout)
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Pipeline did not terminate gracefully, killing")
                    self.process.kill()
                
                self.process = None
            
            self.active = False
            self.logger.info("Pipeline stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping pipeline: {e}")
            return False
    
    def send_frame(self, frame_data: bytes) -> bool:
        """
        Send a frame to the pipeline.
        
        Args:
            frame_data: Raw frame data (bytes)
            
        Returns:
            True if frame was sent, False otherwise
        """
        if not self.active or not self.process or not self.process.stdin:
            return False
        
        try:
            self.process.stdin.write(frame_data)
            return True
        except (BrokenPipeError, IOError) as e:
            self.logger.error(f"Pipe error when sending frame: {e}")
            self.active = False
            return False
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            return False