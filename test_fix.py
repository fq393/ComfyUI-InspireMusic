#!/usr/bin/env python3
# Test script to verify the InspireMusic ComfyUI plugin fixes

import sys
import os
sys.path.append(os.path.dirname(__file__))

from modules.model_manager import InspireMusicModelManager

def test_model_manager():
    """Test if the model manager has the load_model method"""
    print("Testing InspireMusicModelManager...")
    
    # Create model manager instance
    manager = InspireMusicModelManager()
    
    # Check if load_model method exists
    if hasattr(manager, 'load_model'):
        print("✓ load_model method exists")
    else:
        print("✗ load_model method missing")
        return False
    
    # Check available models
    available_models = manager.get_available_models()
    print(f"Available models: {list(available_models.keys())}")
    
    # Test model info retrieval
    for model_name in ["InspireMusic-1.5B-Long", "InspireMusic-1.5B"]:
        info = manager.get_model_info(model_name)
        print(f"Model {model_name} info: {info}")
    
    print("✓ Model manager test completed")
    return True

def test_audio_utils():
    """Test audio utilities"""
    print("\nTesting audio utilities...")
    
    from modules.audio_utils import apply_fade_out, trim_silence, convert_to_comfyui_audio
    import torch
    
    # Create dummy audio
    sample_rate = 24000
    duration = 5.0
    audio = torch.randn(1, int(sample_rate * duration))
    
    # Test apply_fade_out
    try:
        faded_audio = apply_fade_out(audio, sample_rate=sample_rate, fade_duration=1.0)
        print("✓ apply_fade_out works")
    except Exception as e:
        print(f"✗ apply_fade_out failed: {e}")
        return False
    
    # Test trim_silence
    try:
        trimmed_audio = trim_silence(audio, sample_rate=sample_rate)
        print("✓ trim_silence works")
    except Exception as e:
        print(f"✗ trim_silence failed: {e}")
        return False
    
    # Test convert_to_comfyui_audio
    try:
        comfy_audio = convert_to_comfyui_audio(audio, sample_rate)
        print(f"✓ convert_to_comfyui_audio works, format: {type(comfy_audio)}")
    except Exception as e:
        print(f"✗ convert_to_comfyui_audio failed: {e}")
        return False
    
    print("✓ Audio utilities test completed")
    return True

def main():
    print("InspireMusic ComfyUI Plugin Fix Verification")
    print("=" * 50)
    
    success = True
    
    # Test model manager
    success &= test_model_manager()
    
    # Test audio utilities
    success &= test_audio_utils()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! The fixes should resolve the original error.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main()