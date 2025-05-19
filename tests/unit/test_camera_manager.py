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