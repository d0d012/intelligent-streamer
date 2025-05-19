"""
UDP GStreamer pipeline implementation.
"""

from typing import List, Dict, Any, Optional

from .base import GStreamerPipeline

class UdpPipeline(GStreamerPipeline):
    """
    GStreamer pipeline for UDP streaming.
    """
    
    def __init__(
        self,
        source_id: str,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
        host: str = "127.0.0.1",
        port: int = 5000,
        bitrate: int = 800,
        gst_path: str = "gst-launch-1.0"
    ):
        """
        Initialize UDP pipeline.
        
        Args:
            source_id: Unique identifier for the source
            width: Frame width
            height: Frame height
            framerate: Frames per second
            host: Destination host
            port: Destination port
            bitrate: H.264 encoding bitrate in kbps
            gst_path: Path to GStreamer executable
        """
        super().__init__(source_id, gst_path)
        self.width = width
        self.height = height
        self.framerate = framerate
        self.host = host
        self.port = port
        self.bitrate = bitrate
        
    def build_pipeline_args(self) -> List[str]:
        """
        Build the GStreamer pipeline arguments for UDP streaming.
        
        Returns:
            List of command-line arguments for GStreamer
        """
        # Build the pipeline
        args = [
            self.gst_path, "-v",
            "fdsrc", "fd=0",
            "!", f"videoparse", f"width={self.width}", f"height={self.height}",
                 "format=bgr", f"framerate={self.framerate}/1",
            "!", "videoconvert",
            "!", "x264enc", "tune=zerolatency", "speed-preset=superfast", 
                 f"bitrate={self.bitrate}",
            "!", "rtph264pay", "config-interval=1", "pt=96",
            "!", "udpsink", f"host={self.host}", f"port={self.port}"
        ]
        
        return args