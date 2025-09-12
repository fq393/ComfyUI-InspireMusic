# Audio Utilities for InspireMusic ComfyUI Plugin
# Copyright (c) 2024 Alibaba Inc

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional

def load_audio(file_path: str, target_sr: int = 24000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    try:
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            
        return audio, target_sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {file_path}: {str(e)}")

def save_audio(audio: torch.Tensor, file_path: str, sample_rate: int = 24000) -> None:
    """
    Save audio tensor to file.
    
    Args:
        audio: Audio tensor
        file_path: Output file path
        sample_rate: Sample rate
    """
    try:
        # Ensure audio is 2D (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Normalize audio to [-1, 1] range
        audio = torch.clamp(audio, -1.0, 1.0)
        
        torchaudio.save(file_path, audio, sample_rate)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {file_path}: {str(e)}")

def apply_fade_out(audio: torch.Tensor, sample_rate: int = 24000, fade_duration: float = 2.0) -> torch.Tensor:
    """
    Apply fade out effect to audio.
    
    Args:
        audio: Input audio tensor
        fade_duration: Fade duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Audio with fade out applied
    """
    fade_samples = int(fade_duration * sample_rate)
    
    if audio.shape[-1] <= fade_samples:
        return audio
        
    fade_curve = torch.linspace(1.0, 0.0, fade_samples)
    
    # Apply fade to the last fade_samples
    audio_faded = audio.clone()
    audio_faded[..., -fade_samples:] *= fade_curve
    
    return audio_faded

def trim_silence(audio: torch.Tensor, threshold: float = 0.01, sample_rate: int = 24000) -> torch.Tensor:
    """
    Trim silence from the beginning and end of audio.
    
    Args:
        audio: Input audio tensor
        threshold: Silence threshold
        sample_rate: Sample rate
        
    Returns:
        Trimmed audio
    """
    # Calculate RMS energy
    rms = torch.sqrt(torch.mean(audio ** 2, dim=0))
    
    # Find non-silent regions
    non_silent = rms > threshold
    
    if not torch.any(non_silent):
        return audio  # Return original if all silent
        
    # Find start and end indices
    start_idx = torch.argmax(non_silent.float())
    end_idx = len(non_silent) - torch.argmax(torch.flip(non_silent, [0]).float()) - 1
    
    return audio[..., start_idx:end_idx+1]

def convert_to_comfyui_audio(audio: torch.Tensor, sample_rate: int) -> dict:
    """
    Convert audio tensor to ComfyUI audio format.
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        
    Returns:
        Dictionary with ComfyUI audio format
    """
    # Ensure audio is in the right format for ComfyUI
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # ComfyUI expects audio in format (batch, channels, samples)
    if audio.shape[0] == 1:  # If mono
        audio = audio.unsqueeze(0)  # Add batch dimension
    
    return {
        "waveform": audio,
        "sample_rate": sample_rate
    }