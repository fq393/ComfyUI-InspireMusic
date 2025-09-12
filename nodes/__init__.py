# InspireMusic ComfyUI Plugin Nodes
# Copyright (c) 2024 Alibaba Inc

from .inspiremusic_node import (
    InspireMusicTextToMusicNode,
    NODE_CLASS_MAPPINGS as INSPIREMUSIC_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as INSPIREMUSIC_NODE_DISPLAY_NAME_MAPPINGS
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **INSPIREMUSIC_NODE_CLASS_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **INSPIREMUSIC_NODE_DISPLAY_NAME_MAPPINGS
}

__all__ = [
    'InspireMusicTextToMusicNode',
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS'
]