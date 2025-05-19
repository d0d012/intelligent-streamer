"""
Main entry point for the application.
"""

import argparse
import signal
import sys
import time

from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger
from src.video_ingestion.camera_manager import CameraManager

# Global variables for signal handling
camera_manager = None
running = True

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global running
    running = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Intelligent Video Analytics Platform")
    
    # General options
    parser.add_argument("--config", type=str, help="Path to configuration directory")
    parser.add_argument("--env", type=str, help="Environment name (development, production, etc.)")
    
    # Camera options
    parser.add_argument("--webcam", type=int, action="append", help="Add webcam source (can be specified multiple times)")
    parser.add_argument("--rtsp", type=str, action="append", help="Add RTSP source URL (can be specified multiple times)")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    global camera_manager
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment from arguments
    if args.env:
        import os
        os.environ["APP_ENV"] = args.env
    
    # Set up logging
    setup_logging()
    logger = get_logger("main")
    
    logger.info("Starting Intelligent Video Analytics Platform")
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create camera manager
        camera_manager = CameraManager()
        
        # Add sources from command line arguments
        if args.webcam:
            for i, webcam_id in enumerate(args.webcam):
                source_id = camera_manager.add_source(webcam_id, f"webcam_{i}")
                if source_id:
                    logger.info(f"Added webcam {webcam_id} with ID: {source_id}")
                else:
                    logger.error(f"Failed to add webcam {webcam_id}")
        
        if args.rtsp:
            for i, rtsp_url in enumerate(args.rtsp):
                source_id = camera_manager.add_source(rtsp_url, f"rtsp_{i}")
                if source_id:
                    logger.info(f"Added RTSP source {rtsp_url} with ID: {source_id}")
                else:
                    logger.error(f"Failed to add RTSP source {rtsp_url}")
        
        # If no sources were specified, add a default webcam
        if not args.webcam and not args.rtsp:
            logger.info("No sources specified, adding default webcam")
            source_id = camera_manager.add_source(0, "default_webcam")
            if source_id:
                logger.info(f"Added default webcam with ID: {source_id}")
            else:
                logger.error("Failed to add default webcam")
        
        # Start all sources
        if camera_manager.start_all():
            logger.info("Started all sources")
        else:
            logger.error("Failed to start all sources")
        
        # Main loop
        while running:
            # Display status every 5 seconds
            statuses = camera_manager.get_all_statuses()
            
            logger.info(f"Active sources: {sum(1 for s in statuses.values() if s['active'])}/{len(statuses)}")
            
            for source_id, status in statuses.items():
                if status["active"]:
                    logger.info(f"Source {source_id}: FPS={status['fps']:.1f}, Frames={status['frame_count']}")
            
            # Sleep for a while
            time.sleep(5)
        
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    
    finally:
        # Shutdown camera manager
        if camera_manager:
            logger.info("Shutting down camera manager")
            camera_manager.shutdown()
        
        logger.info("Intelligent Video Analytics Platform stopped")

if __name__ == "__main__":
    main()