"""
Unit tests for the configuration system.
"""

import os
import tempfile
import json
import unittest
from pathlib import Path

# Add the src directory to the Python path
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import Config

class TestConfig(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test configs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)
        
        # Create test config files
        default_config = {
            "api": {
                "host": "127.0.0.1",
                "port": 8000
            },
            "analytics": {
                "enabled": True
            }
        }
        
        dev_config = {
            "api": {
                "debug": True
            }
        }
        
        # Write config files
        with open(self.config_dir / "default.json", "w") as f:
            json.dump(default_config, f)
        
        with open(self.config_dir / "development.json", "w") as f:
            json.dump(dev_config, f)
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_load_config(self):
        # Create config with our test directory
        config = Config(config_dir=self.config_dir)
        
        # Test that values were loaded correctly
        self.assertEqual(config.get("api", "host"), "127.0.0.1")
        self.assertEqual(config.get("api", "port"), 8000)
        self.assertTrue(config.get("analytics", "enabled"))
        
        # Test that development values were merged
        self.assertTrue(config.get("api", "debug"))
    
    def test_env_override(self):
        # Set environment variable to override config
        os.environ["APP_API_PORT"] = "9000"
        
        # Create config with our test directory
        config = Config(config_dir=self.config_dir)
        
        # Test that environment variable overrode file config
        self.assertEqual(config.get("api", "port"), 9000)
        
        # Clean up
        del os.environ["APP_API_PORT"]
    
    def test_get_default(self):
        config = Config(config_dir=self.config_dir)
        
        # Test getting non-existent value with default
        self.assertEqual(config.get("nonexistent", "key", "default_value"), "default_value")
    
    def test_get_section(self):
        config = Config(config_dir=self.config_dir)
        
        # Test getting entire section
        api_section = config.get_section("api")
        self.assertIsInstance(api_section, dict)
        self.assertEqual(api_section["host"], "127.0.0.1")
        self.assertEqual(api_section["port"], 8000)
        self.assertTrue(api_section["debug"])

if __name__ == "__main__":
    unittest.main()