"""
ComfyUI NSFW Guard (Viddexa) - Content moderation nodes for ComfyUI

Nodes:
- NSFWCheck: Checks images for NSFW content and interrupts workflow if detected.
  Auto-downloads the model on first use.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

