"""Vision Transformer model definitions.

Public API
----------
``ViTBackbone`` — the core encoder + classification head for DR grading.
All architecture parameters are read from :class:`dr_model.config.Settings`.
"""

from __future__ import annotations

from .backbone import ViTBackbone

__all__ = ["ViTBackbone"]
