#!/usr/bin/env python3

import sys
from pathlib import Path

# Add modules directory to path
sys.path.append('modules')

def test_config_detection():
    """
    Test configuration file detection without importing torch
    """
    print("Testing configuration file detection...")
    
    # Simulate the get_model_config logic without importing the full class
    model_name = "InspireMusic-1.5B-Long"
    model_base_path = Path("models/InspireMusic")
    model_path = model_base_path / model_name
    
    print(f"Model path: {model_path.resolve()}")
    print(f"Model path exists: {model_path.exists()}")
    
    if model_path.exists():
        config_files = [
            "config.json",
            "configuration.json", 
            "inspiremusic.yaml",
            "model_config.json",
            "config.yaml"
        ]
        
        for config_file in config_files:
            config_path = model_path / config_file
            print(f"Checking {config_file}: {config_path.exists()}")
            if config_path.exists():
                print(f"✓ Found config file: {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                        print(f"Config content preview: {content[:100]}...")
                except Exception as e:
                    print(f"Error reading config: {e}")
                return str(config_path)
    
    print("✗ No config file found")
    return None

if __name__ == "__main__":
    test_config_detection()