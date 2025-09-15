# Model Manager for InspireMusic ComfyUI Plugin
# Copyright (c) 2024 Alibaba Inc

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path

class InspireMusicModelManager:
    """
    Manages InspireMusic model loading and configuration.
    """
    
    def __init__(self, model_base_path: str = None):
        """
        Initialize the InspireMusic model manager.
        
        Args:
            model_base_path: Base path where InspireMusic models are stored
        """
        if model_base_path is None:
            # Get the directory of this script and construct the model path
            current_dir = Path(__file__).parent.absolute()
            # Try to find ComfyUI models directory
            # Check if we're in custom_nodes/ComfyUI-InspireMusic structure
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
                    comfyui_root = current_dir.parent.parent
                    model_base_path = comfyui_root / "models" / "InspireMusic"
            else:
                # Fallback to relative path
                model_base_path = current_dir.parent / "models" / "InspireMusic"
        
        self.model_base_path = Path(model_base_path)
        self.loaded_models = {}
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    def get_available_models(self) -> Dict[str, str]:
        """
        Get list of available InspireMusic models.
        
        Returns:
            Dictionary mapping model names to their paths
        """
        models = {}
        
        if not self.model_base_path.exists():
            return models
            
        # Common InspireMusic model names
        model_names = [
            "InspireMusic-1.5B-Long",
            "InspireMusic-1.5B", 
            "InspireMusic-Base",
            "InspireMusic-1.5B-24kHz",
            "InspireMusic-Base-24kHz"
        ]
        
        for model_name in model_names:
            model_path = self.model_base_path / model_name
            if model_path.exists():
                models[model_name] = str(model_path)
                
        return models
        
    def get_model_config(self, model_name: str) -> Optional[str]:
        """
        Get configuration file path for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to config file or None if not found
        """
        # First, try to find config files in the model directory itself
        available_models = self.get_available_models()
        if model_name in available_models:
            model_path = Path(available_models[model_name])
            
            # Look for config files in the model directory
            config_files = [
                "config.json",
                "configuration.json", 
                "inspiremusic.yaml",
                "model_config.json",
                "config.yaml"
            ]
            
            for config_file in config_files:
                config_path = model_path / config_file
                if config_path.exists():
                    return str(config_path)
        
        # Fallback: Map model names to config files in common locations
        config_mapping = {
            "InspireMusic-1.5B-Long": "inspiremusic_1.5b_long.yaml",
            "InspireMusic-1.5B": "inspiremusic_1.5b.yaml",
            "InspireMusic-Base": "inspiremusic.yaml",
            "InspireMusic-1.5B-24kHz": "inspiremusic_1.5b_24khz.yaml",
            "InspireMusic-Base-24kHz": "inspiremusic_24khz.yaml"
        }
        
        config_name = config_mapping.get(model_name)
        if not config_name:
            return None
            
        # Look for config in common locations
        possible_paths = [
            Path("examples/music_generation/conf") / config_name,
            Path("conf") / config_name,
            Path("config") / config_name
        ]
        
        for config_path in possible_paths:
            if config_path.exists():
                return str(config_path)
                
        return None
        
    def get_model_sample_rate(self, model_name: str) -> int:
        """
        Get the default sample rate for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Sample rate in Hz
        """
        if "24kHz" in model_name or "24khz" in model_name.lower():
            return 24000
        else:
            return 48000  # Default for most models
            
    def validate_model_path(self, model_path: str) -> bool:
        """
        Validate if a model path contains required files.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if valid model directory
        """
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            return False
            
        # Check for common model files
        required_patterns = [
            "*.bin",
            "*.safetensors", 
            "config.json",
            "pytorch_model.bin"
        ]
        
        for pattern in required_patterns:
            if list(model_dir.glob(pattern)):
                return True
                
        return False
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        available_models = self.get_available_models()
        
        if model_name not in available_models:
            return {}
            
        model_path = available_models[model_name]
        
        return {
            "name": model_name,
            "path": model_path,
            "config": self.get_model_config(model_name),
            "sample_rate": self.get_model_sample_rate(model_name),
            "valid": self.validate_model_path(model_path),
            "device": self.device
        }
        
    def load_model(self, model_name: str, fast_mode: bool = False, output_sample_rate: int = 48000, model_dir: str = None, max_duration: float = 180.0):
        """
        Load an InspireMusic model.
        
        Args:
            model_name: Name of the model to load
            fast_mode: Whether to use fast inference mode
            output_sample_rate: Target sample rate for output
            model_dir: Optional custom model directory path (for server deployment)
            max_duration: Maximum duration in seconds for audio generation
            
        Returns:
            Loaded InspireMusic model instance
        """
        from inspiremusic.cli.inference import InspireMusicModel
        
        # Check if model is already loaded
        cache_key = f"{model_name}_{fast_mode}_{output_sample_rate}_{model_dir or 'default'}_{max_duration}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
            
        # Use custom model_dir if provided (for server deployment)
        if model_dir:
            model_path = model_dir
        else:
            # Get model information from local paths
            model_info = self.get_model_info(model_name)
            
            if not model_info or not model_info.get('valid', False):
                available_models = list(self.get_available_models().keys())
                raise ValueError(f"Model {model_name} not found or invalid. Available models: {available_models}")
                
            model_path = model_info['path']
            config_path = model_info['config']
            
            if not config_path:
                raise ValueError(f"Configuration file not found for model {model_name}")
            
        try:
            # Load the model using InspireMusic's inference interface
            # For server deployment, let InspireMusic handle the model loading
            model = InspireMusicModel(
                model_name=model_name,
                model_dir=model_path,
                output_sample_rate=output_sample_rate,
                fast=fast_mode,
                gpu=0 if self.device == 'cuda' else -1,
                max_generate_audio_seconds=max_duration  # Use the provided max_duration
            )
            
            # Cache the loaded model
            self.loaded_models[cache_key] = model
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def clear_cache(self):
        """
        Clear loaded model cache.
        """
        self.loaded_models.clear()
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()