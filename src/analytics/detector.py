"""
Object detection models for video analytics.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
import os
import threading

from src.utils.config import get_config
from src.utils.logging import get_logger

class DetectionResult:
    """
    Represents the result of object detection on a frame.
    """
    
    def __init__(self, 
                 detections: List[Dict],
                 frame_id: int = 0, 
                 timestamp: float = None):
        """
        Initialize a detection result.
        
        Args:
            detections: List of detected objects
            frame_id: Frame identifier (optional)
            timestamp: Detection timestamp (default: current time)
        """
        self.detections = detections
        self.frame_id = frame_id
        self.timestamp = timestamp or time.time()
        self.processing_time = 0.0  # Will be set by detector
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "detections": self.detections,
            "count": len(self.detections)
        }
    
    @property
    def count(self) -> int:
        """Get the number of detections."""
        return len(self.detections)


class ObjectDetector(ABC):
    """
    Abstract base class for object detection models.
    """
    
    def __init__(self, 
                confidence_threshold: float = 0.5,
                device: str = "cpu"):
        """
        Initialize detector.
        
        Args:
            confidence_threshold: Minimum confidence for a valid detection
            device: Computation device ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.logger = get_logger(f"detector.{self.__class__.__name__}")
        self.classes = {}  # Will be filled by implementation
        self.model = None  # Will be set by implementation
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the model (load weights, prepare for inference).
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect objects in a frame.
        
        Args:
            frame: BGR or RGB image as numpy array
            
        Returns:
            DetectionResult object with detections
        """
        pass
    
    def annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Original frame
            result: Detection result from detect()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw each detection
        for detection in result.detections:
            # Get bounding box
            bbox = detection["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Get class information
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            # Choose color based on class ID (for consistency)
            color_id = class_id % 6
            colors = [
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 0, 0),    # Blue
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (255, 255, 0)   # Cyan
            ]
            color = colors[color_id]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text = f"{class_name}: {confidence:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated, 
                (x1, y1 - text_size[1] - 5), 
                (x1 + text_size[0], y1), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated, 
                text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        # Add performance info
        cv2.putText(
            annotated,
            f"Detections: {result.count}, Time: {result.processing_time*1000:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated


class YOLODetector(ObjectDetector):
    """
    YOLOv5 implementation for object detection.
    """
    
    def __init__(self, 
                model_name: str = "yolov5s",
                confidence_threshold: float = 0.5,
                device: str = None):
        """
        Initialize YOLOv5 detector.
        
        Args:
            model_name: Model variant (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
            confidence_threshold: Minimum confidence for a valid detection
            device: Computation device (auto-detected if None)
        """
        # Auto-detect device if not specified
        if device is None:
            # Try to detect CUDA availability
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        super().__init__(confidence_threshold, device)
        self.model_name = model_name
        self.logger.info(f"Initializing YOLOv5 detector with {model_name} on {device}")
    
    def initialize(self) -> bool:
        """
        Initialize YOLOv5 model.
        
        Returns:
            True if successful, False otherwise
        """
        if self.initialized:
            return True
        
        try:
            # Import here to make torch optional
            import torch
            import torch.nn as nn
            
            self.logger.info(f"Loading YOLOv5 model: {self.model_name}")
            
            # Load model from torch hub
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                self.model_name, 
                pretrained=True,
                trust_repo=True
            )
            
            # Set parameters
            self.model.conf = self.confidence_threshold  # Confidence threshold
            self.model.iou = 0.45  # IoU threshold for NMS
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # Multiple labels per box
            self.model.max_det = 100  # Maximum detections
            
            # Move to device
            self.model.to(self.device)
            
            # Get class names dictionary
            self.classes = self.model.names
            
            self.logger.info(f"YOLOv5 model loaded successfully with {len(self.classes)} classes")
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOv5: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect objects in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            DetectionResult with detected objects
        """
        if not self.initialized:
            if not self.initialize():
                self.logger.error("Model not initialized")
                return DetectionResult([])
        
        try:
            start_time = time.time()
            
            # Convert BGR to RGB (YOLOv5 expects RGB)
            if frame.shape[2] == 3:  # Make sure it has 3 channels
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Run inference
            results = self.model(rgb_frame)
            
            # Process results to standardized format
            detections = []
            
            # Extract detections from YOLOv5 results
            if len(results.xyxy[0]) > 0:
                # Convert to numpy array for easier processing
                result_array = results.xyxy[0].cpu().numpy()
                
                for x1, y1, x2, y2, conf, class_id in result_array:
                    class_id = int(class_id)
                    detection = {
                        "class_id": class_id,
                        "class_name": self.classes[class_id],
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    }
                    detections.append(detection)
            
            # Create and return result
            processing_time = time.time() - start_time
            result = DetectionResult(detections)
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in YOLOv5 detection: {e}")
            return DetectionResult([])


class DetectorFactory:
    """
    Factory class for creating detector instances.
    """
    
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> ObjectDetector:
        """
        Create a detector instance.
        
        Args:
            detector_type: Type of detector ("yolov5", "ssd", etc.)
            **kwargs: Additional parameters for detector
            
        Returns:
            ObjectDetector instance
        """
        if detector_type.lower() == "yolov5":
            return YOLODetector(**kwargs)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")


class DetectorManager:
    """
    Manager for all detector instances.
    Provides a centralized way to access and manage detectors.
    """
    
    def __init__(self):
        """Initialize detector manager."""
        self.detectors = {}
        self.logger = get_logger("detector_manager")
        self.config = get_config()
        
        # Default model
        self.default_model = self.config.get("analytics", "default_model", "yolov5s")
        self.confidence_threshold = self.config.get("analytics", "confidence_threshold", 0.5)
        
        self.logger.info("Detector manager initialized")
    
    def get_detector(self, detector_id: str = "default") -> ObjectDetector:
        """
        Get or create a detector by ID.
        
        Args:
            detector_id: Detector identifier (default: "default")
            
        Returns:
            ObjectDetector instance
        """
        if detector_id not in self.detectors:
            # Create default detector
            self.logger.info(f"Creating detector: {detector_id}")
            detector = DetectorFactory.create_detector(
                "yolov5",
                model_name=self.default_model,
                confidence_threshold=self.confidence_threshold
            )
            
            # Initialize detector
            if not detector.initialize():
                self.logger.error(f"Failed to initialize detector: {detector_id}")
            
            # Store detector
            self.detectors[detector_id] = detector
        
        return self.detectors[detector_id]
    
    def shutdown(self):
        """Shut down all detectors."""
        self.logger.info("Shutting down detectors")
        self.detectors.clear()
        self.logger.info("All detectors shut down")