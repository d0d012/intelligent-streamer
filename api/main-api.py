from flask import Flask, request, jsonify
import sys
import os
import time

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_ingestion.camera_manager import CameraManager
from src.utils.config import get_config

app = Flask(__name__)  # Fix the syntax error with __name__

# Create camera manager instance
camera_manager = CameraManager()

# Get config
config = get_config()
api_config = config.get_section("api")

@app.route('/streams', methods=['GET'])
def list_streams():
    """Get all available streams"""
    statuses = camera_manager.get_all_statuses()
    return jsonify({"streams": statuses}), 200

@app.route('/streams', methods=['POST'])
def add_stream():
    """Add a new stream"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    # Required parameters
    if 'source' not in data:
        return jsonify({"error": "Source is required"}), 400
    
    # Optional parameters
    source_id = data.get('source_id', None)
    resolution = data.get('resolution', None)
    if resolution and isinstance(resolution, list) and len(resolution) == 2:
        resolution = tuple(resolution)
    framerate = data.get('framerate', None)
    
    # Add the source
    result_id = camera_manager.add_source(
        source=data['source'],
        source_id=source_id,
        resolution=resolution,
        framerate=framerate
    )
    
    if result_id:
        # Get RTSP URLs for this source
        rtsp_urls = camera_manager.get_rtsp_urls(result_id)
        return jsonify({
            "message": f"Stream {result_id} added successfully",
            "stream_id": result_id,
            "rtsp_urls": rtsp_urls.get(result_id, {})
        }), 201
    else:
        return jsonify({"error": "Failed to add stream"}), 500

@app.route('/streams/<string:stream_id>/start', methods=['POST'])
def start_stream(stream_id):
    """Start a specific stream"""
    if camera_manager.get_source(stream_id):
        result = camera_manager.start_source(stream_id)
        if result:
            return jsonify({"message": f"Stream {stream_id} started"}), 200
        else:
            return jsonify({"error": f"Failed to start stream {stream_id}"}), 500
    return jsonify({"error": f"Stream {stream_id} not found"}), 404

@app.route('/streams/<string:stream_id>/stop', methods=['POST'])
def stop_stream(stream_id):
    """Stop a specific stream"""
    if camera_manager.get_source(stream_id):
        result = camera_manager.stop_source(stream_id)
        if result:
            return jsonify({"message": f"Stream {stream_id} stopped"}), 200
        else:
            return jsonify({"error": f"Failed to stop stream {stream_id}"}), 500
    return jsonify({"error": f"Stream {stream_id} not found"}), 404

@app.route('/streams/<string:stream_id>', methods=['DELETE'])
def remove_stream(stream_id):
    """Remove a specific stream"""
    if camera_manager.get_source(stream_id):
        result = camera_manager.remove_source(stream_id)
        if result:
            return jsonify({"message": f"Stream {stream_id} removed"}), 200
        else:
            return jsonify({"error": f"Failed to remove stream {stream_id}"}), 500
    return jsonify({"error": f"Stream {stream_id} not found"}), 404

@app.route('/streams/<string:stream_id>/settings', methods=['POST'])
def adjust_stream_settings(stream_id):
    """Adjust settings for a specific stream"""
    if not camera_manager.get_source(stream_id):
        return jsonify({"error": f"Stream {stream_id} not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    # Prepare settings dict
    settings = {}
    
    if 'resolution' in data:
        if isinstance(data['resolution'], list) and len(data['resolution']) == 2:
            settings['resolution'] = tuple(data['resolution'])
        else:
            return jsonify({"error": "Resolution must be a list of [width, height]"}), 400
    
    if 'framerate' in data:
        settings['framerate'] = data['framerate']

    if settings:
        result = camera_manager.update_source_settings(stream_id, settings)
        if result:
            # Get updated status
            status = camera_manager.get_source_status(stream_id)
            return jsonify({
                "message": f"Settings for stream {stream_id} updated", 
                "settings": status
            }), 200
        else:
            return jsonify({"error": f"Failed to update settings for stream {stream_id}"}), 500
    else:
        return jsonify({"error": "No valid settings provided"}), 400

@app.route('/streams/<string:stream_id>/analytics', methods=['POST'])
def toggle_analytics(stream_id):
    """Enable or disable analytics for a stream"""
    if not camera_manager.get_source(stream_id):
        return jsonify({"error": f"Stream {stream_id} not found"}), 404
    
    data = request.get_json()
    if not data or 'enabled' not in data:
        return jsonify({"error": "Missing 'enabled' parameter"}), 400
    
    enabled = bool(data['enabled'])
    result = camera_manager.toggle_analytics(stream_id, enabled)
    
    if result:
        return jsonify({"message": f"Analytics {'enabled' if enabled else 'disabled'} for stream {stream_id}"}), 200
    else:
        return jsonify({"error": f"Failed to toggle analytics for stream {stream_id}"}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Get overall system status"""
    # Get all stream statuses
    stream_statuses = camera_manager.get_all_statuses()
    
    # Get RTSP URLs
    rtsp_urls = camera_manager.get_rtsp_urls()
    
    return jsonify({
        "system_status": "operational",
        "streams": stream_statuses,
        "rtsp_urls": rtsp_urls
    }), 200

@app.route('/streams/start', methods=['POST'])
def start_all_streams():
    """Start all streams"""
    result = camera_manager.start_all()
    if result:
        return jsonify({"message": "All streams started"}), 200
    else:
        return jsonify({"error": "Failed to start all streams"}), 500

@app.route('/streams/stop', methods=['POST'])
def stop_all_streams():
    """Stop all streams"""
    result = camera_manager.stop_all()
    if result:
        return jsonify({"message": "All streams stopped"}), 200
    else:
        return jsonify({"error": "Failed to stop all streams"}), 500
@app.route('/ping', methods=['GET'])
def ping():
    """Simple endpoint to test if API is running"""
    return jsonify({"status": "online"}), 200

# Shutdown endpoint for graceful termination
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the API server and camera manager"""
    camera_manager.shutdown()
    return jsonify({"message": "System shutdown initiated"}), 200

if __name__ == '__main__':
    # Get port from config
    port = api_config.get("port", 8000)
    host = api_config.get("host", "0.0.0.0")
    debug = api_config.get("debug", False)
    
    # Start the server
    app.run(debug=debug, host=host, port=port)