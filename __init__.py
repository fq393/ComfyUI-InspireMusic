# ComfyUI InspireMusic Plugin
# Copyright (c) 2024 Alibaba Inc

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export the mappings for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Plugin metadata
WEB_DIRECTORY = "./web"
NAME = "ComfyUI-InspireMusic"
VERSION = "1.0.0"
DESCRIPTION = "InspireMusic integration for ComfyUI - AI Music Generation"
AUTHOR = "Alibaba Inc"