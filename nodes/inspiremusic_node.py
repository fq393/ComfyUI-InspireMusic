# Copyright (c) 2024 Alibaba Inc
# ComfyUI InspireMusic Node Implementation

import os
import sys
import tempfile
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from hyperpyyaml import load_hyperpyyaml

# Add inspiremusic to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inspiremusic.cli.inference import InspireMusicModel
from ..modules import (
    load_audio, save_audio, apply_fade_out, 
    trim_silence, convert_to_comfyui_audio,
    InspireMusicModelManager
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InspireMusicTextToMusicNode:
    """ComfyUI Node for InspireMusic Text-to-Music Generation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A captivating classical piano performance with dynamic and intense atmosphere."
                }),
                "model_name": (["InspireMusic-1.5B-Long", "InspireMusic-1.5B", "InspireMusic-Base", "InspireMusic-1.5B-24kHz", "InspireMusic-Base-24kHz"], {
                    "default": "InspireMusic-1.5B-Long"
                }),
                "task_type": (["text-to-music", "continuation"], {
                    "default": "text-to-music"
                }),
                "duration_min": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5
                }),
                "duration_max": ("FLOAT", {
                    "default": 60.0,
                    "min": 5.0,
                    "max": 180.0,
                    "step": 1.0
                }),
                "output_sample_rate": ([24000, 48000], {
                    "default": 48000
                }),
                "chorus_mode": (["default", "random", "verse", "chorus", "intro", "outro"], {
                    "default": "default"
                }),
                "fast_mode": ("BOOLEAN", {
                    "default": False
                }),
                "fade_out": ("BOOLEAN", {
                    "default": True
                }),
                "fade_out_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "trim_silence": ("BOOLEAN", {
                    "default": False
                })
            },
            "optional": {
                "audio_prompt": ("AUDIO", {}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**31 - 1
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)
    CATEGORY = "audio/generation"
    FUNCTION = "generate_music"
    
    def __init__(self):
        self.model = None
        self.model_manager = InspireMusicModelManager()
        self.device = self.model_manager.device
    
    def _load_model(self, model_name: str, fast_mode: bool = False, output_sample_rate: int = 48000):
        """Load the InspireMusic model"""
        try:
            self.model = self.model_manager.load_model(
                model_name=model_name,
                fast_mode=fast_mode,
                output_sample_rate=output_sample_rate
            )
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
            
        return self.model
    
    def _prepare_audio_prompt(self, audio_prompt, output_sample_rate: int):
        """Prepare audio prompt for continuation task"""
        if audio_prompt is None:
            return None
            
        try:
            # Extract audio data from ComfyUI audio format
            if isinstance(audio_prompt, dict) and "waveform" in audio_prompt:
                waveform = audio_prompt["waveform"]
                sample_rate = audio_prompt.get("sample_rate", 24000)
                
                # Convert to numpy and save as temporary file
                import tempfile
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.close()
                
                # Convert tensor to numpy if needed
                if torch.is_tensor(waveform):
                    audio_data = waveform.squeeze().cpu().numpy()
                else:
                    audio_data = waveform
                
                # Ensure audio_data is in correct format (channels, samples)
                if audio_data.ndim == 1:
                    audio_data = audio_data.reshape(1, -1)
                elif audio_data.ndim == 3:  # (batch, channels, samples)
                    audio_data = audio_data.squeeze(0)
                
                # Convert to tensor and save using torchaudio
                audio_tensor = torch.from_numpy(audio_data).float()
                torchaudio.save(temp_file.name, audio_tensor, sample_rate)
                
                return temp_file.name
                
        except Exception as e:
            logging.warning(f"Failed to process audio prompt: {str(e)}")
            return None
        
        return None
    
    def generate_music(self, text_prompt: str, model_name: str, task_type: str,
                      duration_min: float, duration_max: float, output_sample_rate: int,
                      chorus_mode: str, fast_mode: bool, fade_out: bool,
                      fade_out_duration: float, trim_silence: bool,
                      audio_prompt=None, seed: int = -1):
        """Generate music using InspireMusic"""
        
        try:
            # Set seed if provided
            if seed != -1:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Load model
            model = self._load_model(model_name, fast_mode, output_sample_rate)
            
            # Prepare audio prompt if provided
            audio_prompt_path = None
            if audio_prompt is not None and task_type == "continuation":
                audio_prompt_path = self._prepare_audio_prompt(audio_prompt, output_sample_rate)
            
            # Set generation parameters
            time_end = duration_max
            
            # Generate music using the CLI interface
            output_file = model.inference(
                task=task_type,
                text=text_prompt,
                audio_prompt=audio_prompt_path,
                chorus=chorus_mode,
                time_start=0.0,
                time_end=time_end,
                output_fn="temp_output",
                fade_out_duration=fade_out_duration if fade_out else 0.0,
                fade_out_mode=fade_out,
                trim=trim_silence
            )
            
            # Load the generated audio file
            generated_audio = None
            if output_file and os.path.exists(output_file):
                generated_audio, _ = load_audio(output_file, target_sr=output_sample_rate)
                generated_audio = generated_audio.squeeze(0)  # Remove channel dimension
                
                # Clean up the temporary output file
                try:
                    os.unlink(output_file)
                except:
                    pass
            else:
                raise RuntimeError("Failed to generate audio file")
            
            # Ensure generated_audio is not None
            if generated_audio is None:
                raise RuntimeError("Generated audio is None")
            
            # Clean up temporary audio prompt file
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                try:
                    os.unlink(audio_prompt_path)
                except:
                    pass
            
            # Post-processing
            if fade_out:
                generated_audio = apply_fade_out(generated_audio, sample_rate=output_sample_rate, fade_duration=fade_out_duration)
            
            if trim_silence:
                from ..modules.audio_utils import trim_silence as trim_silence_func
                generated_audio = trim_silence_func(generated_audio, sample_rate=output_sample_rate)
            
            # Convert to ComfyUI audio format
            audio_output = convert_to_comfyui_audio(generated_audio, output_sample_rate)
            
            logging.info(f"Successfully generated music with duration: {generated_audio.shape[-1] / output_sample_rate:.2f}s")
            
            return (audio_output,)
            
        except Exception as e:
            logging.error(f"Music generation failed: {str(e)}")
            # Return empty audio on failure
            empty_audio = {
                "waveform": torch.zeros(1, 1, int(output_sample_rate * 5)),
                "sample_rate": output_sample_rate
            }
            return (empty_audio,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "InspireMusicTextToMusic": InspireMusicTextToMusicNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspireMusicTextToMusic": "InspireMusic Text-to-Music Generator"
}