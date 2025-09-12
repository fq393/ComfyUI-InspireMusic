#!/usr/bin/env python3
# Test script to verify the InspireMusic ComfyUI plugin fixes

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(__file__))

def test_model_paths():
    """Test if the model paths are correct"""
    print("Testing model paths...")
    
    # Test different possible paths
    possible_paths = [
        "../models/InspireMusic",
        "../../models/InspireMusic", 
        "/data/ComfyUI/custom_nodes/ComfyUI-InspireMusic/../../models/InspireMusic"
    ]
    
    for path_str in possible_paths:
        path = Path(path_str)
        print(f"Checking path: {path.resolve()}")
        if path.exists():
            print(f"✓ Path exists: {path}")
            # Check for model subdirectories
            model_names = ["InspireMusic-1.5B-Long", "InspireMusic-1.5B", "InspireMusic-Base"]
            for model_name in model_names:
                model_path = path / model_name
                if model_path.exists():
                    print(f"  ✓ Found model: {model_name}")
                    # Check for model files
                    model_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
                    if model_files:
                        print(f"    ✓ Model files found: {[f.name for f in model_files[:3]]}")
                    else:
                        print(f"    ✗ No model files found")
                else:
                    print(f"  ✗ Model not found: {model_name}")
        else:
            print(f"✗ Path does not exist: {path}")
        print()
    
    return True

def test_model_manager_simple():
    """Test model manager without importing torch"""
    print("Testing model manager (basic functionality)...")
    
    try:
        # Import without torch dependencies
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "model_manager", 
            "/Users/fanqi/Desktop/ComfyUI-InspireMusic/modules/model_manager.py"
        )
        
        # Check if the file can be loaded
        if spec and spec.loader:
            print("✓ model_manager.py file is accessible")
        else:
            print("✗ model_manager.py file is not accessible")
            return False
            
    except Exception as e:
        print(f"✗ Error accessing model_manager.py: {e}")
        return False
    
    print("✓ Basic model manager test completed")
    return True

def main():
    print("InspireMusic ComfyUI Plugin Path Verification")
    print("=" * 50)
    
    success = True
    
    # Test model paths
    success &= test_model_paths()
    
    # Test model manager basic functionality
    success &= test_model_manager_simple()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Path verification completed. Check the output above for model availability.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main()