"""
ComfyUI-PreFlight — predict how a generated image/video would be treated by
Instagram, TikTok and X BEFORE publishing (removal vs. reach suppression).

A local Qwen-VL model acts as a pure observation sensor; a deterministic stdlib
rules engine maps those observations to per-platform verdict ranges; a feedback
loop lets you record real outcomes and calibrate the rules over time. This is an
assistive signal, not an approval gate.

ComfyUI imports this package and reads the two mappings below. Uses the classic
(V1) node API for the widest compatibility. Heavy imports (torch/transformers)
are deferred inside the node methods, so this package imports fine in a minimal
"pytest + numpy + Pillow" environment.
"""

try:  # normal ComfyUI package import
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # imported without package context (pytest / tooling)
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
