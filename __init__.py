"""
ComfyUI ARC Save — sign-on-arrival save node for ArcVelvetOS.

Replaces ComfyUI's SaveImage. POSTs the encoded PNG to ArcVelvet's
arcIngest endpoint, receives the C2PA-signed copy, writes the signed
file to the output directory. Fails loudly if the API is unreachable
or the request is rejected — by design, the file must be signed
before it touches disk.

See README.md for setup (API key issuance, arc_config.json) and the
node's wire protocol against arcIngest.
"""

from .nodes.arc_save import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
