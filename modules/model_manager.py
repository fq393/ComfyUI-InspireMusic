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
    
    def __init__(self, model_base_path: str = "../models/InspireMusic"):
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
        # Map model names to config files
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
        
    def load_model(self, model_name: str, fast_mode: bool = False, output_sample_rate: int = 48000):
        """
        Load an InspireMusic model.
        
        Args:
            model_name: Name of the model to load
            fast_mode: Whether to use fast inference mode
            output_sample_rate: Target sample rate for output
            
        Returns:
            Loaded InspireMusic model instance
        """
        from inspiremusic.cli.inference import InspireMusicModel
        
        # Check if model is already loaded
        cache_key = f"{model_name}_{fast_mode}_{output_sample_rate}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
            
        # Get model information
        model_info = self.get_model_info(model_name)
        if not model_info or not model_info.get('valid', False):
            raise ValueError(f"Model {model_name} not found or invalid")
            
        model_path = model_info['path']
        config_path = model_info['config']
        
        if not config_path:
            raise ValueError(f"Configuration file not found for model {model_name}")
            
        try:
            # Load the model using InspireMusic's inference interface
            model = InspireMusicModel(
                model_name=model_name,
                model_dir=model_path,
                output_sample_rate=output_sample_rate,
                fast=fast_mode,
                gpu=0 if self.device == 'cuda' else -1
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