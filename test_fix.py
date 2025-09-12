#!/usr/bin/env python3
"""
Test script to verify the InspireMusic ComfyUI plugin fixes
"""

import sys
import os
from pathlib import Path

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_path_detection():
    """Test path detection logic"""
    print("Testing path detection logic...")
    
    # Simulate different environments
    test_paths = [
        "/Users/fanqi/Desktop/ComfyUI-InspireMusic/modules/model_manager.py",  # Current
        "/data/ComfyUI/custom_nodes/ComfyUI-InspireMusic/modules/model_manager.py",  # Target
    ]
    
    for test_path in test_paths:
        current_dir = Path(test_path).parent.absolute()
        print(f"\nTesting path: {current_dir}")
        
        if "custom_nodes" in str(current_dir):
            # Navigate up to ComfyUI root and find models
            # current_dir is like: /data/ComfyUI/custom_nodes/ComfyUI-InspireMusic/modules
            # We need to go up to /data/ComfyUI/
            # Find the custom_nodes part and go up from there
            path_parts = current_dir.parts
            custom_nodes_index = None
            for i, part in enumerate(path_parts):
                if part == "custom_nodes":
                    custom_nodes_index = i
                    break
            
            if custom_nodes_index is not None:
                # Reconstruct path up to ComfyUI root (before custom_nodes)
                comfyui_root = Path(*path_parts[:custom_nodes_index])
                model_base_path = comfyui_root / "models" / "InspireMusic"
            else:
                # Fallback if we can't find custom_nodes in path
                comfyui_root = current_dir.parent.parent.parent
                model_base_path = comfyui_root / "models" / "InspireMusic"
            print(f"  ComfyUI structure detected")
            print(f"  ComfyUI root: {comfyui_root}")
            print(f"  Model path: {model_base_path}")
        else:
            model_base_path = current_dir.parent / "models" / "InspireMusic"
            print(f"  Fallback structure")
            print(f"  Model path: {model_base_path}")

def test_model_manager():
    """Test the model manager with new path calculation"""
    print("\nTesting Model Manager with new path calculation...")
    
    try:
        from model_manager import InspireMusicModelManager
        
        # Test model manager initialization
        manager = InspireMusicModelManager()
        print(f"✓ Model manager initialized successfully")
        print(f"✓ Model base path: {manager.model_base_path.resolve()}")
        
        # Test getting available models
        available_models = manager.get_available_models()
        print(f"✓ Available models: {list(available_models.keys())}")
        
        # Test with explicit path (simulating ComfyUI environment)
        print("\nTesting with explicit ComfyUI path...")
        comfyui_manager = InspireMusicModelManager("/data/ComfyUI/models/InspireMusic")
        print(f"✓ ComfyUI model manager initialized")
        print(f"✓ ComfyUI model base path: {comfyui_manager.model_base_path.resolve()}")
        
    except Exception as e:
        print(f"✗ Model manager test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("InspireMusic ComfyUI Plugin Path Detection Test")
    print("=" * 55)
    
    test_path_detection()
    test_model_manager()
    
    print("\n" + "=" * 55)
    print("✓ Path detection test completed.")

if __name__ == "__main__":
    main()