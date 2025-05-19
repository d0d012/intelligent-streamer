"""
RTSP GStreamer pipeline implementation with RTSP server.
"""

import os
import subprocess
import signal
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
import tempfile

from .base import GStreamerPipeline
from src.utils.logging import get_logger

class RtspPipeline(GStreamerPipeline):
    """
    GStreamer pipeline for RTSP streaming using rtsp-server.
    """
    
    def __init__(
        self,
        source_id: str,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
        port: int = 8554,
        path: str = None,
        bitrate: int = 2000,
        gst_path: str = "gst-launch-1.0"
    ):
        """
        Initialize RTSP pipeline.
        
        Args:
            source_id: Unique identifier for the source
            width: Frame width
            height: Frame height
            framerate: Frames per second
            port: RTSP server port
            path: RTSP stream path (default: /source_id)
            bitrate: H.264 encoding bitrate in kbps
            gst_path: Path to GStreamer executable
        """
        super().__init__(source_id, gst_path)
        self.width = width
        self.height = height
        self.framerate = framerate
        self.port = port
        self.path = path or f"/{source_id}"
        self.bitrate = bitrate
        
        # RTSP server process
        self.rtsp_server_process = None
        
        # Temporary fifo file for streaming
        self.fifo_path = None
        
        # Logger
        self.logger = get_logger(f"rtsp.{source_id}")
    
    def build_pipeline_args(self) -> List[str]:
        """
        Build the GStreamer pipeline arguments for sending to RTSP server.
        
        Returns:
            List of command-line arguments for GStreamer
        """
        # Create a named pipe (FIFO) for streaming to the RTSP server
        fifo_dir = tempfile.mkdtemp()
        self.fifo_path = os.path.join(fifo_dir, f"rtsp_{self.source_id}.fifo")
        
        # Create the FIFO if it doesn't exist
        if not os.path.exists(self.fifo_path):
            os.mkfifo(self.fifo_path)
        
        self.logger.debug(f"Created FIFO at {self.fifo_path}")
        
        # Build the pipeline that reads from stdin and writes to FIFO
        args = [
            self.gst_path, "-v",
            "fdsrc", "fd=0",
            "!", f"videoparse", f"width={self.width}", f"height={self.height}",
                 "format=bgr", f"framerate={self.framerate}/1",
            "!", "videoconvert",
            "!", "x264enc", "tune=zerolatency", "speed-preset=superfast", 
                 f"bitrate={self.bitrate}", "key-int-max=30",
            "!", "h264parse",
            "!", "mpegtsmux",
            "!", f"filesink", f"location={self.fifo_path}"
        ]
        
        return args
    
    def start(self) -> bool:
        """
        Start the RTSP server and pipeline.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # First, start the RTSP server
            if not self._start_rtsp_server():
                self.logger.error("Failed to start RTSP server")
                return False
            
            # Then start the GStreamer pipeline
            if not super().start():
                self.logger.error("Failed to start GStreamer pipeline")
                self._stop_rtsp_server()
                return False
            
            self.logger.info(f"RTSP server started on port {self.port} with path {self.path}")
            self.logger.info(f"Stream URL: rtsp://localhost:{self.port}{self.path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting RTSP pipeline: {e}")
            self._cleanup()
            return False
    
    def stop(self) -> bool:
        """
        Stop the RTSP server and pipeline.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        success = True
        
        # Stop the GStreamer pipeline
        if not super().stop():
            self.logger.error("Failed to stop GStreamer pipeline")
            success = False
        
        # Stop the RTSP server
        if not self._stop_rtsp_server():
            self.logger.error("Failed to stop RTSP server")
            success = False
        
        # Clean up resources
        self._cleanup()
        
        return success
    
    def _start_rtsp_server(self) -> bool:
        """
        Start the GStreamer RTSP server.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # We'll use test-launch from gst-rtsp-server to create a quick RTSP server
            # On Ubuntu, this is part of the gstreamer1.0-rtsp package
            
            # Common mounting point for all streams
            mount_point = self.path
            
            # The pipeline to serve via RTSP
            pipeline_str = (
                f"( filesrc location={self.fifo_path} ! "
                f"tsdemux ! h264parse ! rtph264pay name=pay0 pt=96 )"
            )
            
            # Command to start the RTSP server
            cmd = [
                "test-launch",
                f"--port={self.port}",
                pipeline_str
            ]
            
            self.logger.debug(f"Starting RTSP server with command: {' '.join(cmd)}")
            
            # Start the server
            self.rtsp_server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it a moment to start
            time.sleep(1)
            
            # Check if it started successfully
            if self.rtsp_server_process.poll() is not None:
                # Process has already exited
                self.logger.error("RTSP server process exited immediately")
                err_output = self.rtsp_server_process.stderr.read().decode('utf-8')
                self.logger.error(f"RTSP server error: {err_output}")
                return False
            
            self.logger.info(f"RTSP server started on port {self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting RTSP server: {e}")
            return False
    
    def _stop_rtsp_server(self) -> bool:
        """
        Stop the RTSP server.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.rtsp_server_process:
            return True
        
        try:
            # Send SIGTERM to the process
            self.rtsp_server_process.terminate()
            
            # Wait for process to end (with timeout)
            try:
                self.rtsp_server_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.logger.warning("RTSP server did not terminate gracefully, killing")
                self.rtsp_server_process.kill()
            
            self.rtsp_server_process = None
            self.logger.info("RTSP server stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping RTSP server: {e}")
            return False
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            # Remove FIFO file
            if self.fifo_path and os.path.exists(self.fifo_path):
                os.unlink(self.fifo_path)
                # Remove the directory if it's empty
                fifo_dir = os.path.dirname(self.fifo_path)
                if os.path.exists(fifo_dir) and not os.listdir(fifo_dir):
                    os.rmdir(fifo_dir)
                self.logger.debug(f"Removed FIFO at {self.fifo_path}")
                self.fifo_path = None
        except Exception as e:
            self.logger.error(f"Error cleaning up FIFO: {e}")