"""
Unit tests for the camera manager.
"""

import sys
import os
import unittest
import time
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.video_ingestion.camera_manager import CameraManager, VideoSource
from src.utils.config import Config

class MockVideoCapture:
    """Mock implementation of OpenCV's VideoCapture."""
    
    def __init__(self, source):
        self.source = source
        self.opened = True
        self.frame_count = 0
    
    def isOpened(self):
        return self.opened
    
    def read(self):
        # Create a simple test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some text to the frame
        cv2.putText(
            frame, 
            f"Test Frame {self.frame_count}", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        self.frame_count += 1
        return True, frame
    
    def set(self, prop, value):
        return True
    
    def release(self):
        self.opened = False


class TestCameraManager(unittest.TestCase):
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    @patch('subprocess.Popen')
    def setUp(self, mock_popen):
        # Mock the subprocess.Popen to avoid actually starting GStreamer
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process
        
        # Create a camera manager
        self.camera_manager = CameraManager()
        
        # Mock config to enable RTSP
        self.camera_manager.streaming_config = {
            "base_port": 5000,
            "rtsp_enabled": True
        }
    
    def test_add_source(self):
        # Test adding a source
        source_id = self.camera_manager.add_source(0, "test_camera")
        
        # Check that the source was added
        self.assertEqual(source_id, "test_camera")
        self.assertIn("test_camera", self.camera_manager.sources)
        
        # Check that the source has the correct properties
        source = self.camera_manager.sources["test_camera"]
        self.assertEqual(source.source, 0)
        self.assertEqual(source.source_id, "test_camera")
        
        # Check that pipelines were created
        self.assertIn("udp", source.raw_pipelines)
        self.assertIn("udp", source.annotated_pipelines)
        self.assertIn("rtsp", source.raw_pipelines)
        self.assertIn("rtsp", source.annotated_pipelines)
    
    def test_start_stop_source(self):
        # Add a source
        source_id = self.camera_manager.add_source(0, "test_camera")
        
        # Start the source
        result = self.camera_manager.start_source(source_id)
        self.assertTrue(result)
        self.assertTrue(self.camera_manager.sources[source_id].active)
        
        # Stop the source
        result = self.camera_manager.stop_source(source_id)
        self.assertTrue(result)
        self.assertFalse(self.camera_manager.sources[source_id].active)
    
    def test_remove_source(self):
        # Add a source
        source_id = self.camera_manager.add_source(0, "test_camera")
        
        # Remove the source
        result = self.camera_manager.remove_source(source_id)
        self.assertTrue(result)
        self.assertNotIn(source_id, self.camera_manager.sources)
    
    def test_start_stop_all(self):
        # Add multiple sources
        self.camera_manager.add_source(0, "camera1")
        self.camera_manager.add_source(1, "camera2")
        
        # Start all sources
        result = self.camera_manager.start_all()
        self.assertTrue(result)
        
        # Check that all sources are active
        for source in self.camera_manager.sources.values():
            self.assertTrue(source.active)
        
        # Stop all sources
        result = self.camera_manager.stop_all()
        self.assertTrue(result)
        
        # Check that all sources are inactive
        for source in self.camera_manager.sources.values():
            self.assertFalse(source.active)
    
    def test_get_source_status(self):
        # Add a source
        source_id = self.camera_manager.add_source(0, "test_camera")
        
        # Get status
        status = self.camera_manager.get_source_status(source_id)
        
        # Check status properties
        self.assertEqual(status["source_id"], source_id)
        self.assertFalse(status["active"])
        self.assertEqual(status["resolution"], (640, 480))
        self.assertEqual(status["framerate"], 30)
    
    def test_toggle_analytics(self):
        # Add a source
        source_id = self.camera_manager.add_source(0, "test_camera")
        
        # By default, analytics should be enabled
        status = self.camera_manager.get_source_status(source_id)
        self.assertTrue(status["analytics_enabled"])
        
        # Disable analytics
        result = self.camera_manager.toggle_analytics(source_id, False)
        self.assertTrue(result)
        
        # Check that analytics is disabled
        status = self.camera_manager.get_source_status(source_id)
        self.assertFalse(status["analytics_enabled"])
    
    def test_update_source_settings(self):
        # Add a source
        source_id = self.camera_manager.add_source(0, "test_camera")
        
        # Update settings
        new_settings = {
            "resolution": (1280, 720),
            "framerate": 60
        }
        result = self.camera_manager.update_source_settings(source_id, new_settings)
        self.assertTrue(result)
        
        # Check that settings were updated
        source = self.camera_manager.get_source(source_id)
        self.assertEqual(source.resolution, (1280, 720))
        self.assertEqual(source.framerate, 60)
    
    def test_get_all_statuses(self):
        # Add multiple sources
        self.camera_manager.add_source(0, "camera1")
        self.camera_manager.add_source(1, "camera2")
        
        # Get all statuses
        statuses = self.camera_manager.get_all_statuses()
        
        # Check that we have the correct number of statuses
        self.assertEqual(len(statuses), 2)
        self.assertIn("camera1", statuses)
        self.assertIn("camera2", statuses)


class TestVideoSource(unittest.TestCase):
    
    @patch('cv2.VideoCapture', MockVideoCapture)
    @patch('subprocess.Popen')
    def test_video_source_lifecycle(self, mock_popen):
        # Mock the subprocess.Popen to avoid actually starting GStreamer
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process
        
        # Create a video source
        source = VideoSource(
            source_id="test",
            source=0,
            resolution=(640, 480),
            framerate=30
        )
        
        # Add pipelines
        from src.video_ingestion.pipelines.udp import UdpPipeline
        
        udp_pipeline = UdpPipeline(
            source_id="test",
            width=640,
            height=480,
            framerate=30,
            port=5000
        )
        
        source.add_raw_pipeline("udp", udp_pipeline)
        
        # Start the source
        result = source.start()
        self.assertTrue(result)
        self.assertTrue(source.active)
        
        # Let it run for a moment to process some frames
        time.sleep(0.1)
        
        # Get status
        status = source.get_status()
        self.assertEqual(status["source_id"], "test")
        self.assertTrue(status["active"])
        
        # Stop the source
        result = source.stop()
        self.assertTrue(result)
        self.assertFalse(source.active)


if __name__ == "__main__":
    unittest.main()